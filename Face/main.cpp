#include <QCoreApplication>
#include <QCommandLineParser>
#include <QFileInfo>
#include <QTextStream>
#include <QDateTime>
#include <thread>
#include <chrono>
#include <QDir>
#include <QFileInfo>
#include <opencv2/opencv.hpp>

using namespace cv;

static QString makeOutputPath(const QString& basePath, const QString& resultTag) {
    // basePath가 디렉토리면 timestamp 파일명, 아니면 그대로 사용
    QFileInfo fi(basePath);
    if (fi.isDir()) {
        const auto now = QDateTime::currentDateTime();
        const QString ts = now.toString("yyyyMMdd_hhmmss");
        return QDir(basePath).filePath(QString("%1_%2.jpg").arg(ts, resultTag));
    }
    // 부모 디렉토리가 없으면 생성
    QDir dir = fi.dir();
    if (!dir.exists()) dir.mkpath(".");
    return fi.absoluteFilePath();
}

static bool saveImage(const cv::Mat& img, const QString& path, QTextStream& out, QTextStream& err) {
    QFileInfo fi(path);
    QDir dir = fi.dir();
    if (!dir.exists() && !dir.mkpath(".")) {
        err << "[ERR] Cannot create directory: " << dir.absolutePath() << Qt::endl;
        return false;
    }
    const std::string p = fi.absoluteFilePath().toStdString();
    bool ok = cv::imwrite(p, img);
    out << "[SAVE] " << fi.absoluteFilePath() << (ok ? "  (ok)" : "  (FAILED)") << Qt::endl;
    return ok;
}

static bool loadCascade(CascadeClassifier& cc, const QString& path, const char* name, QTextStream& err) {
    if (path.isEmpty() || !QFileInfo::exists(path)) {
        err << "[ERR] " << name << " cascade not found: " << path << Qt::endl;
        return false;
    }
    if (!cc.load(path.toStdString())) {
        err << "[ERR] Failed to load " << name << " cascade: " << path << Qt::endl;
        return false;
    }
    return true;
}

static bool detectEyesNoseMouth(const Mat& bgr,
                                CascadeClassifier& faceCC,
                                CascadeClassifier& eyesCC,
                                CascadeClassifier& noseCC,
                                CascadeClassifier& mouthCC,
                                Mat* annotatedOut = nullptr)
{
    Mat gray;
    cvtColor(bgr, gray, COLOR_BGR2GRAY);
    equalizeHist(gray, gray);

    std::vector<Rect> faces;
    faceCC.detectMultiScale(gray, faces, 1.1, 3, CASCADE_SCALE_IMAGE, Size(80, 80));

    for (const Rect& f : faces) {
        Mat faceROI = gray(f);

        Rect upper(faceROI.cols * 0.05, faceROI.rows * 0.10,
                   faceROI.cols * 0.90, faceROI.rows * 0.45);
        Rect mid  (faceROI.cols * 0.15, faceROI.rows * 0.35,
                 faceROI.cols * 0.70, faceROI.rows * 0.40);
        Rect lower(faceROI.cols * 0.10, faceROI.rows * 0.55,
                   faceROI.cols * 0.80, faceROI.rows * 0.40);
        Rect bounds(0,0,faceROI.cols,faceROI.rows);
        upper &= bounds; mid &= bounds; lower &= bounds;

        // Eyes (>=2)
        std::vector<Rect> eyes;
        eyesCC.detectMultiScale(faceROI(upper), eyes, 1.1, 3, CASCADE_SCALE_IMAGE, Size(16,16));
        int eyeCount = 0;
        for (const Rect& e : eyes) if (e.width>=14 && e.height>=14) eyeCount++;
        bool okEyes = (eyeCount >= 2);

        // Nose (>=1)
        std::vector<Rect> noses;
        noseCC.detectMultiScale(faceROI(mid), noses, 1.1, 3, CASCADE_SCALE_IMAGE, Size(18,18));
        bool okNose = !noses.empty();

        // Mouth (>=1) — mouth 또는 smile cascade
        std::vector<Rect> mouths;
        mouthCC.detectMultiScale(faceROI(lower), mouths, 1.2, 10, CASCADE_SCALE_IMAGE, Size(22,14));
        bool okMouth = !mouths.empty();

        if (okEyes && okNose && okMouth) {
            if (annotatedOut) {
                Mat vis; cvtColor(gray, vis, COLOR_GRAY2BGR); // 원본 bgr 복사도 가능
                vis = bgr.clone();
                rectangle(vis, f, Scalar(0,255,0), 2);
                for (const Rect& e : eyes) {
                    Rect r(f.x + upper.x + e.x, f.y + upper.y + e.y, e.width, e.height);
                    rectangle(vis, r, Scalar(255,0,0), 2);
                }
                for (const Rect& n : noses) {
                    Rect r(f.x + mid.x + n.x, f.y + mid.y + n.y, n.width, n.height);
                    rectangle(vis, r, Scalar(0,255,255), 2);
                }
                for (const Rect& m : mouths) {
                    Rect r(f.x + lower.x + m.x, f.y + lower.y + m.y, m.width, m.height);
                    rectangle(vis, r, Scalar(0,0,255), 2);
                }
                *annotatedOut = std::move(vis);
            }
            return true;
        }
    }
    return false;
}

int main(int argc, char *argv[]) {
    QCoreApplication app(argc, argv);
    QCoreApplication::setApplicationName("rpi_facial_parts_monitor");
    QCoreApplication::setApplicationVersion("2.0");

    QCommandLineParser p;
    p.setApplicationDescription("Raspberry Pi camera monitor: every N seconds, check both eyes + nose + mouth (infinite loop).");
    p.addHelpOption();
    p.addVersionOption();

    // 입력 선택: libcamera(GStreamer) 또는 V4L2
    QCommandLineOption useLibcameraOpt("use-libcamera", "Use libcamera (GStreamer pipeline).");
    QCommandLineOption pipelineOpt("pipeline", "Custom GStreamer pipeline for libcamera.", "PIPE");
    QCommandLineOption camIndexOpt({"d","device"}, "V4L2 camera index (ignored if --use-libcamera).", "INDEX", "0");

    // 카메라 파라미터
    QCommandLineOption widthOpt("w", "Capture width.", "W", "640");
    //QCommandLineOption heightOpt("h", "Capture height.", "H", "480");
    QCommandLineOption heightOpt("height", "Capture height.", "H", "480");
    QCommandLineOption fpsOpt("fps", "Capture FPS.", "FPS", "30");
    QCommandLineOption warmupOpt("warmup", "Warm-up frames at start.", "N", "5");

    // 모니터링 주기/윈도우
    QCommandLineOption intervalOpt("interval", "Seconds between checks.", "SECS", "5");
    QCommandLineOption windowOpt("window", "Per-check detection window seconds.", "SECS", "2");

    // Cascade 경로
    QCommandLineOption faceOpt("face", "Face cascade path.", "PATH",
                               "/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt2.xml");
    QCommandLineOption eyesOpt("eyes", "Eyes cascade path.", "PATH",
                               "/usr/share/opencv4/haarcascades/haarcascade_eye_tree_eyeglasses.xml");
    QCommandLineOption noseOpt("nose", "Nose cascade path.", "PATH",
                               "/usr/share/opencv4/haarcascades/haarcascade_mcs_nose.xml");
    QCommandLineOption mouthOpt("mouth", "Mouth/Smile cascade path.", "PATH",
                                "/usr/share/opencv4/haarcascades/haarcascade_smile.xml");

    // 결과 이미지 저장(성공 시 최신 1장만 덮어쓰기)
    QCommandLineOption saveOpt("save", "Save annotated frame on success to this path.", "PATH");
    // 실패했을 때 저장 옵션 (--save-fail)
    QCommandLineOption saveFailOpt(
        "save-fail",
        "Save frame on FAIL cycles (debug). "
        "If PATH is a directory, file will be timestamped.",
        "PATH"
        );
    p.addOption(saveFailOpt);
    p.addOption(useLibcameraOpt);
    p.addOption(pipelineOpt);
    p.addOption(camIndexOpt);
    p.addOption(widthOpt);
    p.addOption(heightOpt);
    p.addOption(fpsOpt);
    p.addOption(warmupOpt);
    p.addOption(intervalOpt);
    p.addOption(windowOpt);
    p.addOption(faceOpt);
    p.addOption(eyesOpt);
    p.addOption(noseOpt);
    p.addOption(mouthOpt);
    p.addOption(saveOpt);
    p.process(app);

    QTextStream out(stdout), err(stderr);

    // Cascade 로드
    CascadeClassifier faceCC, eyesCC, noseCC, mouthCC;
    if (!loadCascade(faceCC, p.value(faceOpt),  "face",  err))  return 1;
    if (!loadCascade(eyesCC, p.value(eyesOpt),  "eyes",  err))  return 1;
    if (!loadCascade(noseCC, p.value(noseOpt),  "nose",  err))  return 1;
    if (!loadCascade(mouthCC, p.value(mouthOpt), "mouth", err)) return 1;

    // 카메라 열기
    int W = p.value(widthOpt).toInt();
    int H = p.value(heightOpt).toInt();
    int FPS = p.value(fpsOpt).toInt();
    int warmupN = std::max(0, p.value(warmupOpt).toInt());

    VideoCapture cap;
    if (p.isSet(useLibcameraOpt)) {
        QString pipe = p.value(pipelineOpt);
        if (pipe.isEmpty()) {
            pipe = QString("libcamerasrc ! video/x-raw, width=%1, height=%2, framerate=%3/1 "
                           "! videoconvert ! appsink").arg(W).arg(H).arg(FPS);
        }
        if (!cap.open(pipe.toStdString(), cv::CAP_GSTREAMER)) {
            err << "[ERR] Failed to open libcamera pipeline: " << pipe << Qt::endl;
            return 1;
        }
    } else {
        int index = p.value(camIndexOpt).toInt();
        if (!cap.open(index, cv::CAP_V4L2)) {
            err << "[ERR] Failed to open V4L2 device index " << index << Qt::endl;
            return 1;
        }
        cap.set(CAP_PROP_FRAME_WIDTH,  W);
        cap.set(CAP_PROP_FRAME_HEIGHT, H);
        cap.set(CAP_PROP_FPS, FPS);
    }

    // 워밍업
    Mat frame;
    for (int i=0; i<warmupN; ++i) cap.read(frame);

    const int intervalSec = std::max(1, p.value(intervalOpt).toInt()); // 기본 5
    const double windowSec = std::max(0.2, p.value(windowOpt).toDouble()); // 기본 2.0
    const bool doSave = p.isSet(saveOpt);
    const QString savePath = p.value(saveOpt);

    out << "[INFO] Started monitoring. interval=" << intervalSec
        << "s, window=" << windowSec << "s. Press Ctrl+C to stop." << Qt::endl;

    // ===== 무한 루프 =====
    while (true) {
        const auto cycleStart = std::chrono::steady_clock::now();
        bool detected = false;
        Mat annotated;

        // windowSec 동안 여러 프레임 검사
        const int64 t0 = cv::getTickCount();
        const double freq = cv::getTickFrequency();
        while (((cv::getTickCount() - t0) / freq) < windowSec) {
            if (!cap.read(frame) || frame.empty()) continue;
            if (detectEyesNoseMouth(frame, faceCC, eyesCC, noseCC, mouthCC, doSave ? &annotated : nullptr)) {
                detected = true;
                break;
            }
            // 너무 바쁘면 살짝 쉼(라즈베리파이 CPU 배려)
            cv::waitKey(1);
        }

        const auto now = QDateTime::currentDateTime();
        // if (detected) {
        //     out << now.toString("yyyy-MM-dd hh:mm:ss") << "  RESULT: OK" << Qt::endl;
        //     if (doSave && !annotated.empty()) {
        //         imwrite(savePath.toStdString(), annotated);
        //     }
        // } else {
        //     out << now.toString("yyyy-MM-dd hh:mm:ss") << "  RESULT: FAIL" << Qt::endl;
        // }
        if (detected) {
            out << now.toString("yyyy-MM-dd hh:mm:ss") << "  RESULT: OK" << Qt::endl;
            if (p.isSet(saveOpt)) {
                QString outPath = makeOutputPath(p.value(saveOpt), "OK");
                saveImage(annotated.empty() ? frame : annotated, outPath, out, err);
            }
        } else {
            out << now.toString("yyyy-MM-dd hh:mm:ss") << "  RESULT: FAIL" << Qt::endl;
            if (p.isSet(saveFailOpt)) {
                QString outPath = makeOutputPath(p.value(saveFailOpt), "FAIL");
                saveImage(frame, outPath, out, err);
            }
        }
        out.flush();

        // 다음 주기까지 대기(주기성 유지)
        const auto cycleEnd = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(cycleEnd - cycleStart);
        auto sleepMs = intervalSec*1000 - (int)elapsed.count();
        if (sleepMs > 0) std::this_thread::sleep_for(std::chrono::milliseconds(sleepMs));
    }

    // 도달하지 않음
    // return 0;
}
