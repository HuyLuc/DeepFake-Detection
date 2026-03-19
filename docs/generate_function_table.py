"""
Tạo Bảng 4.1: Danh sách các chức năng hệ thống
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import textwrap
import os

# --- Dữ liệu bảng ---
headers = ["Mã\nchức năng", "Tên chức năng", "Mô tả chi tiết"]

data = [
    ["FC01", "Phát hiện deepfake\ntrên ảnh/video",
     "Người dùng tải ảnh (PNG, JPG, JPEG, GIF, BMP, WEBP) hoặc video\n"
     "(MP4, AVI, MOV, MKV, WEBM, FLV) lên hệ thống qua giao diện\n"
     "kéo-thả. Chọn 1 trong 3 chế độ: Standard (EfficientNet-B4),\n"
     "Advanced (EfficientNet-B4 + BiLSTM), Ensemble (kết hợp cả hai).\n"
     "Kết quả trả về: FAKE / SUSPICIOUS / REAL kèm xác suất."],

    ["FC02", "Sinh bản đồ nhiệt\nGrad-CAM (XAI)",
     "Khi bật tùy chọn \"Generate Heatmap\", hệ thống sử dụng thuật\n"
     "toán Grad-CAM nhắm vào lớp conv_head của EfficientNet-B4 để\n"
     "tạo bản đồ nhiệt trực quan, cho biết vùng nào trên khuôn mặt\n"
     "mà mô hình tập trung phân tích khi đưa ra quyết định."],

    ["FC03", "Quản lý lịch sử\nphân tích",
     "Tự động lưu toàn bộ kết quả vào bảng prediction_history\n"
     "(SQLite). Người dùng xem lại lịch sử bao gồm: tên tệp, loại\n"
     "tệp, mô hình đã dùng, phán định, xác suất, thời gian xử lý,\n"
     "số khung hình đã phân tích và ảnh thu nhỏ (thumbnail)."],

    ["FC04", "Xuất báo cáo\nPDF / JSON",
     "Xuất kết quả phân tích dưới dạng PDF (ReportLab, A4, chuyên\n"
     "nghiệp với tiêu đề, bảng, heatmap) hoặc JSON (dữ liệu thô\n"
     "có cấu trúc phục vụ tích hợp hoặc lưu trữ)."],

    ["FC05", "Chatbot tư vấn\nAI (RAG)",
     "Tích hợp Google Gemini 2.5 Flash qua API. Chatbot dùng phương\n"
     "pháp RAG – kết hợp knowledge base nội bộ về deepfake với ngữ\n"
     "cảnh hệ thống (thống kê, tin tức) để trả lời câu hỏi chính\n"
     "xác và có căn cứ. Ưu tiên trả lời từ KB trước khi gọi LLM."],

    ["FC06", "Nguồn tin tức\ndeepfake",
     "Module NewsService tự động thu thập và hiển thị các bài viết,\n"
     "nghiên cứu mới nhất liên quan đến deepfake. Trang /news trình\n"
     "bày tin tức dưới dạng danh sách có tóm tắt."],

    ["FC07", "Demo tạo\ndeepfake",
     "Trang demo minh họa các phương pháp tạo deepfake: face_swap,\n"
     "expression_transfer, aging, attribute_edit. Chỉ mang tính chất\n"
     "giáo dục giúp người dùng hiểu bản chất kỹ thuật deepfake."],
]

# --- Cấu hình ---
fig_width = 16
row_heights = [0.6, 1.2, 0.95, 0.85, 0.8, 1.2, 0.75, 0.75]  # header + 7 rows
col_widths = [0.09, 0.14, 0.77]
total_height = sum(row_heights)

fig, ax = plt.subplots(figsize=(fig_width, total_height))
ax.set_xlim(0, 1)
ax.set_ylim(0, total_height)
ax.axis('off')

# Màu sắc
header_color = '#1a365d'
header_text_color = 'white'
row_colors = ['#f7fafc', '#edf2f7']
border_color = '#cbd5e0'
code_color = '#2b6cb0'

# --- Vẽ bảng ---
y_pos = total_height  # Bắt đầu từ trên

for row_idx in range(-1, len(data)):  # -1 = header
    rh = row_heights[row_idx + 1]
    y_pos -= rh

    if row_idx == -1:
        # Header
        x = 0
        for col_idx, w in enumerate(col_widths):
            rect = mpatches.FancyBboxPatch(
                (x, y_pos), w, rh,
                boxstyle="square,pad=0",
                facecolor=header_color, edgecolor=border_color, linewidth=1.2
            )
            ax.add_patch(rect)
            ax.text(x + w / 2, y_pos + rh / 2, headers[col_idx],
                    ha='center', va='center', fontsize=12, fontweight='bold',
                    color=header_text_color, fontfamily='serif')
            x += w
    else:
        bg = row_colors[row_idx % 2]
        x = 0
        for col_idx, w in enumerate(col_widths):
            rect = mpatches.FancyBboxPatch(
                (x, y_pos), w, rh,
                boxstyle="square,pad=0",
                facecolor=bg, edgecolor=border_color, linewidth=0.8
            )
            ax.add_patch(rect)

            cell_text = data[row_idx][col_idx]

            if col_idx == 0:
                # Mã chức năng - bold, màu xanh
                ax.text(x + w / 2, y_pos + rh / 2, cell_text,
                        ha='center', va='center', fontsize=11,
                        fontweight='bold', color=code_color, fontfamily='monospace')
            elif col_idx == 1:
                ax.text(x + w / 2, y_pos + rh / 2, cell_text,
                        ha='center', va='center', fontsize=10.5,
                        fontweight='bold', color='#1a202c', fontfamily='serif')
            else:
                ax.text(x + 0.012, y_pos + rh / 2, cell_text,
                        ha='left', va='center', fontsize=9.5,
                        color='#2d3748', fontfamily='serif', linespacing=1.45)
            x += w

# Viền ngoài
outer_rect = mpatches.FancyBboxPatch(
    (0, 0), 1, total_height,
    boxstyle="square,pad=0",
    facecolor='none', edgecolor=header_color, linewidth=2
)
ax.add_patch(outer_rect)

# Tiêu đề bảng
fig.text(0.5, 0.995, "Bảng 4.1: Danh sách các chức năng hệ thống",
         ha='center', va='top', fontsize=14, fontweight='bold',
         fontfamily='serif', color='#1a202c')

plt.tight_layout()
plt.subplots_adjust(top=0.975)

out_dir = os.path.join(os.path.dirname(__file__), '..', 'evaluation_results', 'report_charts')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'bang_4_1_chuc_nang_he_thong.png')
fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white', pad_inches=0.15)
plt.close()
print(f"Saved: {os.path.abspath(out_path)}")
