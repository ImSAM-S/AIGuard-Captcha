# AIGuard Captcha (FaunaGuard) 🛡️🐾

AIGuard Captcha là một hệ thống xác thực bảo mật thông minh, kết hợp giữa mạng nơ-ron sâu (Deep Learning) và các thuật toán học máy truyền thống để phân loại hành vi người dùng và ngăn chặn Bot. Dự án sử dụng bộ dữ liệu động vật đa dạng để tạo ra các thử thách nhận diện hình ảnh chính xác.

# 🚀 Tính năng nổi bật
Hybrid AI Model: Kết hợp CNN (MobileNetV2), KNN và Random Forest để tối ưu hóa độ chính xác.

High Performance: Sử dụng PCA để giảm chiều dữ liệu, đảm bảo tốc độ phản hồi dưới 1 giây.

Security: Tích hợp cơ chế mã hóa dữ liệu trao đổi giữa Client và Server.

Scalability: Kiến trúc sẵn sàng chuyển đổi sang TensorFlow Lite cho các thiết bị di động.

# 🛠️ Công nghệ sử dụng
Ngôn ngữ: Python 3.x

Deep Learning: TensorFlow, Keras (MobileNetV2)

Machine Learning: Scikit-learn (KNN, Random Forest, IncrementalPCA)

Backend: Flask / FastAPI

Frontend: HTML5, CSS3, JavaScript

Công cụ: GitHub, Discord, Canva (Design)

# 📂 Cấu trúc thư mục
Ai_Captcha/
  ├── data/                                  /Thư mục chứa tập dữ liệu Animals-10
  ├── captcha_animal_model.h5                /Lưu trữ file model đã huấn luyện
  ├── static/                                /Các file CSS, JS, hình ảnh giao diện
  ├── templates/                             /Giao diện HTML
  ├── app.py                                 /Backend chính xử lý logic và API
  ├── train.py                               /Script huấn luyện và đánh giá mô hình
  └── README.md                              /Tài liệu hướng dẫn dự án
# 📊 Kết quả huấn luyện
Hệ thống được huấn luyện trên tập dữ liệu 20.213 ảnh (Animals-10) với các chỉ số hiệu năng ấn tượng:

CNN Acc: ~95% (Nhận diện hình ảnh tối ưu)

Random Forest Acc: ~89% (Độ ổn định cao)

KNN Acc: ~82% (Tốc độ xử lý nhanh)

# 🔧 Cài đặt và Sử dụng
Clone repository:

Bash
git clone https://github.com/ImSAM-S/AIGuard-Captcha.git
cd AIGuard-Captcha
Cài đặt môi trường:

Bash
pip install -r requirements.txt
Huấn luyện mô hình (Tùy chọn):

Bash
python train.py
Chạy ứng dụng:

Bash
python app.py
Sau đó truy cập http://127.0.0.1:5000 trên trình duyệt.


# 📜 Mục Đích
Dự án được phát triển cho mục đích học thuật và nghiên cứu bảo mật
