# Hệ thống cảnh báo buồn ngủ

## Phát hiện buồn ngủ là một công nghệ an toàn có thể ngăn ngừa tai nạn do người lái ngủ gật khi đang điều khiển xe.

Mục tiêu của dự án Python này là xây dựng một hệ thống phát hiện buồn ngủ, có khả năng nhận biết khi mắt người dùng đóng trong một vài giây. Hệ thống sẽ cảnh báo người lái ngay khi phát hiện dấu hiệu buồn ngủ.

### Trong dự án Python này, chúng ta sẽ sử dụng OpenCV để thu ảnh từ webcam và đưa vào mô hình Deep Learning để phân loại xem mắt người dùng đang ở trạng thái “Open” hay “Closed”. Phương pháp được sử dụng trong dự án này gồm các bước sau:

Bước 1 – Thu ảnh đầu vào từ camera.

Bước 2 – Phát hiện khuôn mặt trong ảnh và tạo vùng quan tâm (ROI).

Bước 3 – Phát hiện mắt trong ROI và đưa vào bộ phân loại.

Bước 4 – Bộ phân loại sẽ phân loại xem mắt đang mở hay đóng.

Bước 5 – Tính điểm để kiểm tra xem người dùng có đang buồn ngủ hay không.

### Để xem tài liệu đầy đủ về dự án, vui lòng bấm [vào đây](https://docs.google.com/document/d/1cbtvioQ2iUIOZpgaq-6xQQshJ0zT-vl2/edit?usp=drive_link&ouid=109982577522846244381&rtpof=true&sd=true)
