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
#include <QTcpSocket>
#include <opencv2/dnn.hpp>
#include <cmath>

using namespace cv;
using namespace cv::dnn;
static QTcpSocket tcp;

// 나이 버킷과 중간값(기대나이 계산용)
static const std::vector<std::string> AGE_BUCKETS = {
    "(0-2)","(4-6)","(8-12)","(15-20)","(25-32)","(38-43)","(48-53)","(60-100)"
};
static const std::vector<float> AGE_MIDPOINTS = {
    1.f, 5.f, 10.f, 18.f, 28.5f, 40.5f, 50.5f, 80.f
};
// Caffe age-net 평균값 (BGR)
static const Scalar AGE_MODEL_MEAN(78.4263, 87.7689, 114.8958);

static bool ensureConnected(const QString& host, quint16 port) {
    if (tcp.state() == QAbstractSocket::ConnectedState) return true;
    // 끊겨 있으면 재시도
    if (tcp.state() != QAbstractSocket::UnconnectedState) {
        tcp.abort(); // 즉시 끊고
    }
    tcp.connectToHost(host, port);
    if (!tcp.waitForConnected(2000)) {
        qWarning() << "[TCP] connect failed:" << tcp.errorString()
        << " state=" << tcp.state();
        return false;
    }
    qInfo() << "[TCP] connected to" << host << ":" << port;
    return true;
}

// ====== 결과 전송할 때 이렇게 호출 ======
static void sendResultWithAge(bool maskOK, bool hasAge, float expAge, int ageBucket, float conf, float ageThreshold)
{
    const QString host = "192.168.2.49";   // ← 네가 쓰는 호스트와 동일하게
    const quint16 port = 9000;

    if (!ensureConnected(host, port)) return;

    // 레이블: threshold 기준으로 elder/young, 없으면 unknown
    const char* ageLabel = hasAge ? ((expAge >= ageThreshold) ? "elder" : "young") : "unknown";

    // "1,elder\n" 또는 "0,unknown\n"
    QByteArray payload = QByteArray::asprintf("%d,%s\n", maskOK ? 1 : 0, ageLabel);

    if (tcp.write(payload) == -1) {
        qWarning() << "[TCP] write failed:" << tcp.errorString();
        tcp.abort();
        return;
    }
    tcp.flush();
    tcp.waitForBytesWritten(200);
}

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
                                Mat* annotatedOut = nullptr,
                                Rect* faceRectOut = nullptr)
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
            //추가됨.
            if(faceRectOut) *faceRectOut = f;
            return true;
        }
    }
    return false;
}

//나이값 로더 추가
static bool loadAgeNet(Net& net, const QString& proto, const QString& model, QTextStream& err) {
    if (proto.isEmpty() || !QFileInfo::exists(proto)) {
        err << "[ERR] age-proto not found: " << proto << Qt::endl;
        return false;
    }
    if (model.isEmpty() || !QFileInfo::exists(model)) {
        err << "[ERR] age-model not found: " << model << Qt::endl;
        return false;
    }
    net = readNetFromCaffe(proto.toStdString(), model.toStdString());
    if (net.empty()) {
        err << "[ERR] failed to load age net" << Qt::endl;
        return false;
    }
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);
    return true;
}

// 얼굴 ROI(bgr)로 기대나이/버킷/신뢰도 계산
static bool estimateAge(const Mat& faceBGR, Net& ageNet,
                        float& expectedAge, int& bucketIdx, float& conf)
{
    // 모델 입력 크기: 227x227, BGR 평균값 적용
    Mat blob = blobFromImage(faceBGR, 1.0, Size(227,227), AGE_MODEL_MEAN, false, false);
    ageNet.setInput(blob);
    Mat prob = ageNet.forward();   // 1x8

    Mat row = prob.reshape(1,1);
    double s = cv::sum(row)[0];
    if (s > 0) row /= s;

    Point classId;
    double maxProb;
    minMaxLoc(row, nullptr, &maxProb, nullptr, &classId);
    bucketIdx = classId.x;
    conf = static_cast<float>(maxProb);

    expectedAge = 0.f;
    for (int i=0; i<(int)AGE_BUCKETS.size(); ++i) {
        expectedAge += row.at<float>(0,i) * AGE_MIDPOINTS[i];
    }
    return true;
}

// 추가된 status overlay
static void drawStatusOverlay(cv::Mat& img, const cv::Rect& face,
                              bool maskOK,
                              bool hasAge, float expAge, int ageBucket, float conf,
                              float ageThreshold)
{
    // 박스 색상: 마스크 OK면 초록, 아니면 빨강
    Scalar boxColor = maskOK ? Scalar(0,255,0) : Scalar(0,0,255);
    rectangle(img, face, boxColor, 2);

    // 라벨 텍스트 구성
    char line1[128];  // 마스크 상태
    snprintf(line1, sizeof(line1), "MASK: %s", maskOK ? "OK" : "FAIL");

    char line2[128] = {0}; // 나이 결과
    if (hasAge) {
        // AGE_BUCKETS, expAge, conf 사용
        snprintf(line2, sizeof(line2), "AGE: %s",
                 ageBucket<4?"young":"elder");
    } else {
        snprintf(line2, sizeof(line2), "AGE: N/A");
    }

    // 라벨 영역 배경 박스
    int base=0;
    Size s1 = getTextSize(line1, FONT_HERSHEY_SIMPLEX, 0.6, 2, &base);
    Size s2 = getTextSize(line2, FONT_HERSHEY_SIMPLEX, 0.6, 2, &base);
    int w = std::max(s1.width, s2.width) + 16;
    int h = s1.height + s2.height + 22;

    int y = std::max(face.y - 12, h + 5);
    rectangle(img,
              Point(face.x, y - h),
              Point(face.x + w, y),
              Scalar(0,0,0), FILLED);

    // 텍스트 그리기
    putText(img, line1, Point(face.x + 8, y - h/2 - 4),
            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255,255,255), 2);
    putText(img, line2, Point(face.x + 8, y - 6),
            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255,255,255), 2);

    // 나이 임계 이상이면 얼굴 테두리에 추가 강조(파란 테두리)
    if (hasAge && expAge >= ageThreshold) {
        rectangle(img, face, Scalar(255,0,0), 2);
    }
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
    QCommandLineOption saveOpt("save", "Save annotated frame on success to this path.", "PATH",".");
    // 실패했을 때 저장 옵션 (--save-fail)
    QCommandLineOption saveFailOpt(
        "save-fail",
        "Save frame on FAIL cycles (debug). "
        "If PATH is a directory, file will be timestamped.",
        "PATH"
        );
    // yang 모델 경로/ 임계값
    QCommandLineOption ageProtoOpt("age-proto", "Caffe age deploy prototxt path.", "PATH","/home/kosa/models/age_deploy.prototxt");
    QCommandLineOption ageModelOpt("age-model", "Caffe age caffemodel path.", "PATH","/home/kosa/models/age_net.caffemodel");
    QCommandLineOption ageThreshOpt("age-threshold", "Age threshold (>= this → OK).", "N", "40");

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
    // 추가됨 age
    p.addOption(ageProtoOpt);
    p.addOption(ageModelOpt);
    p.addOption(ageThreshOpt);

    p.process(app);

    QTextStream out(stdout), err(stderr);

    tcp.connectToHost("192.168.1.57", 9000);   // 수신 서버 IP/포트
    tcp.waitForConnected(2000);

    // Cascade 로드
    CascadeClassifier faceCC, eyesCC, noseCC, mouthCC;
    if (!loadCascade(faceCC, p.value(faceOpt),  "face",  err))  return 1;
    if (!loadCascade(eyesCC, p.value(eyesOpt),  "eyes",  err))  return 1;
    if (!loadCascade(noseCC, p.value(noseOpt),  "nose",  err))  return 1;
    if (!loadCascade(mouthCC, p.value(mouthOpt), "mouth", err)) return 1;

    // ---- (카메라 열고 워밍업 전) Age Net 로드 ----
    Net ageNet;
    bool useAge = false;
    float ageThreshold = p.value(ageThreshOpt).toFloat();   // 기본 50

    // 추가됨 for ageproto
    const QString ageProtoPath = p.value(ageProtoOpt);   // 기본값/인자 모두 반영됨
    const QString ageModelPath = p.value(ageModelOpt);

    if (!ageProtoPath.isEmpty() && !ageModelPath.isEmpty()) {
        out << "[INFO] age-proto=" << ageProtoPath << Qt::endl;
        out << "[INFO] age-model=" << ageModelPath << Qt::endl;

    // if (p.isSet(ageProtoOpt) && p.isSet(ageModelOpt)) {
    //    if (loadAgeNet(ageNet, p.value(ageProtoOpt), p.value(ageModelOpt), err)) {
        if (loadAgeNet(ageNet, ageProtoPath, ageModelPath, err)) {
            useAge = true;
            out << "[INFO] AgeNet loaded. threshold=" << ageThreshold << Qt::endl;
        } else {
            out << "[WARN] AgeNet not loaded. Age condition disabled." << Qt::endl;
        }
    } else {
        out << "[WARN] --age-proto/--age-model not provided. Age condition disabled." << Qt::endl;
    }

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
    //savePath
    const QString savePath = p.value(saveOpt);
    const bool doSave = !savePath.isEmpty();


    out << "[INFO] Started monitoring. interval=" << intervalSec
        << "s, window=" << windowSec << "s. Press Ctrl+C to stop." << Qt::endl;

    // ===== 무한 루프 =====
    while (true) {
        const auto cycleStart = std::chrono::steady_clock::now();
        bool detected = false;
        Mat annotated;
        // 추가됨
        Rect faceRect;
        // windowSec 동안 여러 프레임 검사
        const int64 t0 = cv::getTickCount();
        const double freq = cv::getTickFrequency();
        while (((cv::getTickCount() - t0) / freq) < windowSec) {
            if (!cap.read(frame) || frame.empty()) continue;
            // if (detectEyesNoseMouth(frame, faceCC, eyesCC, noseCC, mouthCC, doSave ? &annotated : nullptr)) {
            //     detected = true;
            //     break;
            // }
            //detect 판정 들어간다.
            bool partsOK = detectEyesNoseMouth(frame, faceCC, eyesCC, noseCC, mouthCC,
                                               (p.isSet(saveOpt) ? &annotated : nullptr),
                                               &faceRect);

            if (partsOK) {
                // 기본은 마스크 OK
                detected = true;

                bool hasAge = false;
                float expAge = 0.f, conf = 0.f;
                int   idx = 0;

                if (useAge) {
                    // 얼굴 ROI 추출 (경계보정)
                    Rect imgBounds(0,0,frame.cols, frame.rows);
                    Rect safe = faceRect & imgBounds;
                    if (safe.width > 0 && safe.height > 0) {
                        Mat faceROI = frame(safe).clone();

                        if (estimateAge(faceROI, ageNet, expAge, idx, conf)) {
                            // 디버그 출력
                            hasAge=true;
                            out << "[RESULT] MASK=OK, [AGE]= " << AGE_BUCKETS[idx].c_str()
                                << " (~" << (int)std::lround(expAge)
                                << ")  conf=" << QString::number(conf,'f',2)
                                << Qt::endl;
                        }
                    }
                } else {
                    // AgeNet 비활성화면 기존 로직과 동일하게 눈코입 OK만으로 성공 처리
                    out << "[RESULT] MASK=OK, AGE=N/A" << Qt::endl;
                }

                // 화면 표기 이미지 대상 선택
                Mat& vis = (annotated.empty() ? frame : annotated);
                drawStatusOverlay(vis, faceRect, /*maskOK*/true,
                                  hasAge, expAge, idx, conf,
                                  ageThreshold);

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
            if (doSave) {
                QString outPath = makeOutputPath(savePath, "OK");
                saveImage(annotated.empty() ? frame : annotated, outPath, out, err);
            }
            sendResultWithAge(/*maskOK*/true, hasAge, expAge, idx, conf, ageThreshold);
        } else {
            out << now.toString("yyyy-MM-dd hh:mm:ss") << "  RESULT: FAIL" << Qt::endl;
            if (p.isSet(saveFailOpt)) {
                QString outPath = makeOutputPath(p.value(saveFailOpt), "FAIL");
                saveImage(frame, outPath, out, err);
            }
            sendResultWithAge(/*maskOK*/false, /*hasAge*/false, /*expAge*/0.f, /*idx*/0, /*conf*/0.f, ageThreshold);
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
