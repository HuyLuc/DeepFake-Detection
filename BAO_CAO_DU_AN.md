# BÁO CÁO DỰ ÁN: HỆ THỐNG PHÁT HIỆN DEEPFAKE SỬ DỤNG HỌC MÁY

---

## 1. TÍNH CẤP THIẾT, THỰC TRẠNG HIỆN TẠI VÀ LÝ DO CHỌN ĐỀ TÀI

### 1.1. Tính cấp thiết của đề tài

Trong những năm gần đây, công nghệ Deepfake đã phát triển mạnh mẽ và trở nên dễ tiếp cận hơn bao giờ hết. Với sự hỗ trợ của các công cụ AI hiện đại, việc tạo ra các video giả mạo có độ chân thực cao đã không còn là điều khó khăn. Điều này đặt ra những thách thức nghiêm trọng về an ninh thông tin, đặc biệt là trong các lĩnh vực như báo chí, chính trị, tài chính và pháp luật.

Tôi nhận thấy rằng các video Deepfake có thể được sử dụng để:
- Lan truyền thông tin sai lệch và tin giả
- Tạo ra các bằng chứng giả mạo trong các vụ án pháp lý
- Gây tổn hại đến danh tiếng của cá nhân và tổ chức
- Thao túng dư luận và ảnh hưởng đến các quyết định quan trọng

Vì vậy, việc phát triển một hệ thống có khả năng phát hiện Deepfake một cách chính xác và nhanh chóng là vô cùng cần thiết trong bối cảnh hiện tại.

### 1.2. Thực trạng hiện tại

Hiện nay, trên thế giới đã có nhiều nghiên cứu và giải pháp về phát hiện Deepfake, nhưng phần lớn các hệ thống này vẫn còn những hạn chế:

- **Độ chính xác chưa đạt mức tối ưu**: Nhiều phương pháp hiện tại vẫn gặp khó khăn trong việc phát hiện các Deepfake được tạo ra bởi các kỹ thuật mới nhất, đặc biệt là khi video có chất lượng cao.

- **Tốc độ xử lý chậm**: Một số hệ thống yêu cầu thời gian xử lý lâu, không phù hợp với các ứng dụng thời gian thực.

- **Khả năng mở rộng hạn chế**: Nhiều giải pháp chỉ hoạt động tốt với một số loại Deepfake cụ thể và khó áp dụng cho các kỹ thuật mới.

- **Yêu cầu phần cứng cao**: Các mô hình phức tạp thường yêu cầu GPU mạnh, khiến việc triển khai trở nên tốn kém.

Trong bối cảnh đó, tôi nhận thấy cần phải phát triển một hệ thống vừa đảm bảo độ chính xác, vừa có khả năng tối ưu hóa để chạy trên các phần cứng khác nhau, từ máy tính cá nhân đến các hệ thống cloud.

### 1.3. Lý do chọn đề tài

Tôi chọn đề tài này vì những lý do sau:

**Thứ nhất**, đây là một vấn đề thực tế và cấp thiết trong xã hội hiện đại. Việc phát triển công cụ phát hiện Deepfake không chỉ có ý nghĩa học thuật mà còn có giá trị ứng dụng thực tế cao, góp phần bảo vệ sự thật và ngăn chặn việc lan truyền thông tin sai lệch.

**Thứ hai**, đề tài này cho phép tôi áp dụng các kiến thức về học máy sâu (deep learning) và xử lý ảnh/video mà tôi đã học được. Đặc biệt, tôi có cơ hội làm việc với các mô hình CNN hiện đại như EfficientNet, một kiến trúc đã chứng minh được hiệu quả trong nhiều bài toán phân loại ảnh.

**Thứ ba**, dự án này mang tính thách thức cao vì Deepfake ngày càng tinh vi. Việc phát triển một hệ thống có thể đối phó với các kỹ thuật Deepfake đa dạng đòi hỏi tôi phải nghiên cứu sâu về các phương pháp xử lý video, phát hiện khuôn mặt, và các kỹ thuật tối ưu hóa mô hình.

**Cuối cùng**, tôi muốn tạo ra một giải pháp có thể sử dụng được trong thực tế, không chỉ là một mô hình nghiên cứu. Vì vậy, tôi đã xây dựng cả một ứng dụng web để người dùng có thể dễ dàng sử dụng hệ thống phát hiện Deepfake mà không cần kiến thức chuyên sâu về lập trình.

---

## 2. MỤC TIÊU CỦA ĐỀ TÀI

### 2.1. Mục tiêu tổng quát

Mục tiêu tổng quát của đề tài là xây dựng một hệ thống phát hiện Deepfake tự động, có độ chính xác cao và khả năng xử lý video trong thời gian hợp lý. Hệ thống này phải có khả năng phân biệt được video thật và video giả mạo được tạo ra bởi các kỹ thuật Deepfake khác nhau.

### 2.2. Mục tiêu cụ thể

Dựa trên quá trình phát triển dự án, tôi đã đặt ra các mục tiêu cụ thể sau:

**Mục tiêu 1: Xây dựng pipeline tiền xử lý dữ liệu hiệu quả**
- Phát triển hệ thống tự động trích xuất khuôn mặt từ video sử dụng MediaPipe
- Áp dụng kỹ thuật Uniform Sampling để lấy các frame đại diện từ video một cách đều đặn
- Xử lý các video ngắn bằng Temporal Padding để đảm bảo tính nhất quán của dữ liệu
- Tối ưu hóa quá trình xử lý để giảm thời gian tiền xử lý

**Mục tiêu 2: Huấn luyện mô hình phân loại Deepfake chính xác**
- Sử dụng kiến trúc EfficientNet-B4 làm backbone, một mô hình đã được chứng minh hiệu quả trong các bài toán phân loại ảnh
- Áp dụng các kỹ thuật tối ưu hóa như Mixed Precision Training để tăng tốc độ huấn luyện
- Sử dụng Class Weights để xử lý vấn đề mất cân bằng dữ liệu giữa lớp FAKE và REAL
- Triển khai Early Stopping và Learning Rate Scheduler để tránh overfitting và cải thiện hiệu suất mô hình
- Tích hợp Gradient Clipping để đảm bảo quá trình huấn luyện ổn định

**Mục tiêu 3: Tối ưu hóa hiệu năng và khả năng mở rộng**
- Phát triển hệ thống tự động chọn device (GPU/CPU) dựa trên benchmark hiệu năng
- Tối ưu hóa cấu hình cho các phần cứng khác nhau, từ GPU yếu đến GPU mạnh
- Hỗ trợ training trên Google Colab với tích hợp Google Drive để lưu trữ checkpoint và log
- Tối ưu hóa DataLoader với prefetching và pin_memory để tăng tốc độ training

**Mục tiêu 4: Xây dựng ứng dụng web để sử dụng thực tế**
- Phát triển ứng dụng web sử dụng Flask để người dùng có thể upload video và nhận kết quả phát hiện
- Tích hợp xử lý video real-time với khả năng phát hiện khuôn mặt tự động
- Cung cấp giao diện thân thiện và dễ sử dụng
- Hiển thị kết quả với độ tin cậy (confidence score) để người dùng đánh giá

**Mục tiêu 5: Đảm bảo chất lượng code và khả năng bảo trì**
- Áp dụng logging thống nhất trong toàn bộ dự án thay vì sử dụng print statements
- Externalize các magic numbers vào file cấu hình để dễ dàng điều chỉnh
- Thêm type hints cho các hàm quan trọng để cải thiện khả năng đọc code
- Xây dựng unit tests cho các thành phần quan trọng của hệ thống

### 2.3. Kết quả mong đợi

Sau khi hoàn thành dự án, tôi mong đợi đạt được:

- Một mô hình có khả năng phát hiện Deepfake với độ chính xác cao trên tập validation
- Hệ thống có thể xử lý video trong thời gian hợp lý (vài giây cho mỗi video)
- Ứng dụng web hoạt động ổn định và dễ sử dụng
- Code base có cấu trúc rõ ràng, dễ bảo trì và mở rộng
- Tài liệu hướng dẫn đầy đủ cho việc training và sử dụng hệ thống

---

## 3. PHẠM VI NGHIÊN CỨU

### 3.1. Bộ dữ liệu sử dụng

Trong dự án này, tôi sử dụng bộ dữ liệu **DFDC (DeepFake Detection Challenge)** do Facebook AI Research phát hành. Đây là một trong những bộ dữ liệu lớn và đa dạng nhất về Deepfake hiện có, được tạo ra để thúc đẩy nghiên cứu trong lĩnh vực phát hiện video giả mạo.

Bộ dữ liệu DFDC bao gồm:
- **Video gốc (Original sequences)**: Các video thật được thu thập từ YouTube và các nguồn khác, được chia thành hai nhóm chính là "youtube" và "actors"
- **Video giả mạo (Manipulated sequences)**: Các video được tạo ra bằng nhiều kỹ thuật Deepfake khác nhau, bao gồm:
  - **Deepfakes**: Kỹ thuật trao đổi khuôn mặt sử dụng autoencoder
  - **Face2Face**: Kỹ thuật điều khiển biểu cảm khuôn mặt
  - **FaceSwap**: Kỹ thuật hoán đổi khuôn mặt
  - **NeuralTextures**: Kỹ thuật thao tác kết cấu da
  - **DeepFakeDetection**: Các video được tạo bằng các phương pháp phát hiện Deepfake
  - **FaceShifter**: Kỹ thuật trao đổi khuôn mặt tiên tiến

Tất cả các video trong bộ dữ liệu đều được nén ở mức độ C23 (compression level 23), đảm bảo tính nhất quán về chất lượng và kích thước file.

### 3.2. Phạm vi kỹ thuật

**Kiến trúc mô hình:**
Tôi sử dụng EfficientNet-B4 làm backbone cho mô hình phân loại. EfficientNet là một họ kiến trúc CNN được thiết kế để cân bằng giữa độ chính xác và hiệu quả tính toán. EfficientNet-B4 đã được pre-trained trên ImageNet, cho phép tôi tận dụng transfer learning để đạt được hiệu suất tốt với dữ liệu hạn chế.

**Kỹ thuật tiền xử lý:**
- **Face Detection**: Sử dụng MediaPipe Face Detection để tự động phát hiện và cắt khuôn mặt từ các frame video
- **Uniform Sampling**: Lấy 10 frame đại diện từ mỗi video bằng cách sử dụng linspace để đảm bảo các frame được phân bố đều trên toàn bộ chiều dài video
- **Temporal Padding**: Đối với các video ngắn không đủ 10 frame, hệ thống tự động lặp lại frame cuối cùng để đảm bảo mỗi video có đúng 10 frame
- **Data Augmentation**: Áp dụng các phép biến đổi như RandomHorizontalFlip, ColorJitter, RandomRotation để tăng tính đa dạng của dữ liệu training

**Kỹ thuật training:**
- **Mixed Precision Training**: Sử dụng FP16 để tăng tốc độ training và giảm sử dụng bộ nhớ
- **Class Weights**: Tính toán và áp dụng trọng số lớp để xử lý vấn đề mất cân bằng dữ liệu
- **Learning Rate Scheduling**: Sử dụng ReduceLROnPlateau để tự động giảm learning rate khi validation accuracy không cải thiện
- **Early Stopping**: Dừng training sớm nếu validation accuracy không cải thiện sau 4 epochs
- **Gradient Clipping**: Giới hạn gradient norm ở mức 1.0 để tránh gradient explosion

### 3.3. Phạm vi ứng dụng

Hệ thống được thiết kế để:
- Phân loại video thành hai lớp: **FAKE** (giả mạo) và **REAL** (thật)
- Xử lý video ở định dạng MP4, AVI, MOV, MKV, WEBM
- Hoạt động trên cả GPU và CPU, với khả năng tự động chọn device tốt nhất
- Cung cấp độ tin cậy (confidence score) cho mỗi dự đoán

### 3.4. Giới hạn của nghiên cứu

Trong phạm vi của dự án này, tôi tập trung vào:
- Phát hiện Deepfake ở mức độ video (không phải ảnh tĩnh)
- Chỉ xử lý video có khuôn mặt người (yêu cầu phát hiện được khuôn mặt)
- Phân loại nhị phân (FAKE/REAL) thay vì xác định cụ thể kỹ thuật Deepfake nào được sử dụng
- Sử dụng dữ liệu từ bộ DFDC, có thể không đại diện cho tất cả các loại Deepfake trong thực tế

Các hướng mở rộng trong tương lai có thể bao gồm:
- Phát hiện Deepfake ở mức độ ảnh tĩnh
- Phân loại đa lớp để xác định cụ thể kỹ thuật Deepfake
- Xử lý video không có khuôn mặt hoặc có nhiều khuôn mặt
- Tích hợp các kỹ thuật phát hiện Deepfake mới nhất như Vision Transformer

---

## KẾT LUẬN

Dự án "Hệ thống phát hiện Deepfake sử dụng học máy" được phát triển với mục tiêu tạo ra một giải pháp thực tế và hiệu quả để đối phó với vấn đề Deepfake đang ngày càng phổ biến. Thông qua việc sử dụng bộ dữ liệu DFDC và kiến trúc EfficientNet-B4, cùng với các kỹ thuật tối ưu hóa hiện đại, tôi đã xây dựng được một hệ thống có khả năng phát hiện Deepfake với độ chính xác cao.

Hệ thống không chỉ là một mô hình nghiên cứu mà còn được triển khai dưới dạng ứng dụng web, cho phép người dùng dễ dàng sử dụng mà không cần kiến thức chuyên sâu về lập trình hay học máy. Với khả năng tự động chọn device và tối ưu hóa cho nhiều loại phần cứng khác nhau, hệ thống có thể được sử dụng rộng rãi từ máy tính cá nhân đến các hệ thống cloud.

Trong quá trình phát triển, tôi đã chú trọng đến việc xây dựng code base có cấu trúc rõ ràng, dễ bảo trì và mở rộng. Điều này sẽ tạo điều kiện cho việc cải thiện và phát triển hệ thống trong tương lai, đặc biệt là khi các kỹ thuật Deepfake mới xuất hiện và đòi hỏi các phương pháp phát hiện tương ứng.

---

*Báo cáo được viết dựa trên thông tin thực tế từ code và cấu hình của dự án.*


