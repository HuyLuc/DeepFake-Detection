"""
Tạo Hình 4.1: Sơ đồ Use Case - Ứng dụng web phát hiện Deepfake
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import os

fig, ax = plt.subplots(figsize=(16, 13))
ax.set_xlim(-1, 17)
ax.set_ylim(-1.5, 13.5)
ax.axis('off')
ax.set_aspect('equal')

# === Màu sắc ===
actor_color = '#2b6cb0'
uc_fill = '#ebf8ff'
uc_border = '#2b6cb0'
system_border = '#1a365d'
extend_color = '#e53e3e'
include_color = '#38a169'
line_color = '#4a5568'

# === Vẽ System Boundary ===
sys_rect = FancyBboxPatch(
    (3.5, -0.8), 12, 13.5,
    boxstyle="round,pad=0.3",
    facecolor='#f7fafc', edgecolor=system_border,
    linewidth=2.5, linestyle='-', zorder=0
)
ax.add_patch(sys_rect)
ax.text(9.5, 12.4, "Hệ thống Web Phát hiện Deepfake",
        ha='center', va='center', fontsize=14, fontweight='bold',
        color=system_border, fontfamily='serif',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=system_border, linewidth=1.5))


# === Vẽ Actor ===
def draw_actor(ax, x, y, label, color=actor_color):
    # Head
    head = plt.Circle((x, y + 1.45), 0.3, fill=False, edgecolor=color, linewidth=2, zorder=5)
    ax.add_patch(head)
    # Body
    ax.plot([x, x], [y + 1.15, y + 0.5], color=color, linewidth=2, zorder=5)
    # Arms
    ax.plot([x - 0.4, x + 0.4], [y + 0.9, y + 0.9], color=color, linewidth=2, zorder=5)
    # Legs
    ax.plot([x, x - 0.35], [y + 0.5, y - 0.1], color=color, linewidth=2, zorder=5)
    ax.plot([x, x + 0.35], [y + 0.5, y - 0.1], color=color, linewidth=2, zorder=5)
    # Label
    ax.text(x, y - 0.45, label, ha='center', va='top', fontsize=11,
            fontweight='bold', color=color, fontfamily='serif')


# Vẽ 2 actors
draw_actor(ax, 1.5, 6.5, "Người dùng")
draw_actor(ax, 1.5, 1.5, "Quản trị viên", color='#805ad5')


# === Vẽ Use Case (ellipse) ===
def draw_usecase(ax, cx, cy, text, w=2.8, h=0.75):
    ellipse = mpatches.Ellipse(
        (cx, cy), w, h,
        facecolor=uc_fill, edgecolor=uc_border,
        linewidth=1.8, zorder=3
    )
    ax.add_patch(ellipse)
    ax.text(cx, cy, text, ha='center', va='center', fontsize=9.5,
            fontweight='bold', color='#1a365d', fontfamily='serif', zorder=4)
    return (cx, cy)


# --- Use Cases chính (Người dùng) ---
uc_upload  = draw_usecase(ax, 7,  11.2, "Tải ảnh/video lên")
uc_select  = draw_usecase(ax, 7,   9.6, "Chọn chế độ\nphân tích")
uc_detect  = draw_usecase(ax, 7,   8.0, "Phân tích\nDeepfake")
uc_result  = draw_usecase(ax, 7,   6.4, "Xem kết quả\n& Grad-CAM")
uc_history = draw_usecase(ax, 7,   4.8, "Xem lịch sử\nphân tích")
uc_export  = draw_usecase(ax, 7,   3.2, "Xuất báo cáo\nPDF / JSON")
uc_chat    = draw_usecase(ax, 7,   1.6, "Chat với\nChatbot AI")
uc_news    = draw_usecase(ax, 7,   0.0, "Xem tin tức\nDeepfake")

# --- Use Cases mở rộng (bên phải) ---
uc_standard  = draw_usecase(ax, 12.5, 10.8, "Standard\n(EfficientNet-B4)", w=3.0)
uc_advanced  = draw_usecase(ax, 12.5,  9.3, "Advanced\n(EfficientNet+LSTM)", w=3.0)
uc_ensemble  = draw_usecase(ax, 12.5,  7.8, "Ensemble\n(Kết hợp)", w=3.0)

uc_heatmap   = draw_usecase(ax, 12.5,  6.2, "Sinh bản đồ\nnhiệt Grad-CAM", w=3.0)
uc_face      = draw_usecase(ax, 12.5,  4.6, "Phát hiện &\ncắt khuôn mặt", w=3.0)

uc_pdf       = draw_usecase(ax, 12.5,  3.0, "Xuất PDF\n(ReportLab)", w=3.0)
uc_json      = draw_usecase(ax, 12.5,  1.6, "Xuất JSON", w=3.0)

uc_rag       = draw_usecase(ax, 12.5,  0.0, "RAG: Knowledge\nBase + Gemini", w=3.0)


# === Vẽ đường nối ===
def draw_line(ax, x1, y1, x2, y2, color=line_color, lw=1.2, style='-'):
    ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, linestyle=style, zorder=2)


def draw_arrow_label(ax, x1, y1, x2, y2, label, color='#718096', style='--', lw=1.0):
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw, linestyle=style),
                zorder=2)
    ax.text(mx + 0.15, my + 0.2, label, ha='center', va='center', fontsize=8,
            fontstyle='italic', color=color,
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.9),
            zorder=4)


# Người dùng → các UC chính
for uc_pos in [uc_upload, uc_select, uc_detect, uc_result, uc_history, uc_export, uc_chat, uc_news]:
    draw_line(ax, 2.1, 7.5, uc_pos[0] - 1.4, uc_pos[1])

# Quản trị viên → một số UC
for uc_pos in [uc_history, uc_export, uc_news]:
    draw_line(ax, 2.1, 2.5, uc_pos[0] - 1.4, uc_pos[1], color='#805ad5', lw=1.0)

# <<extend>> : Chọn chế độ → 3 chế độ
draw_arrow_label(ax, 8.4, 10.8, 11.0, 10.8, "«extends»", extend_color, '--')
draw_arrow_label(ax, 8.4,  9.6, 11.0,  9.3, "«extends»", extend_color, '--')
draw_arrow_label(ax, 8.4,  9.0, 11.0,  7.8, "«extends»", extend_color, '--')

# <<include>> : Phân tích → Detect face
draw_arrow_label(ax, 8.4, 7.8, 11.0, 4.6, "«include»", include_color, '--')

# <<extend>> : Xem kết quả → Grad-CAM
draw_arrow_label(ax, 8.4, 6.4, 11.0, 6.2, "«extends»", extend_color, '--')

# <<extend>> : Xuất báo cáo → PDF, JSON
draw_arrow_label(ax, 8.4, 3.4, 11.0, 3.0, "«extends»", extend_color, '--')
draw_arrow_label(ax, 8.4, 3.0, 11.0, 1.6, "«extends»", extend_color, '--')

# <<include>> : Chat → RAG
draw_arrow_label(ax, 8.4, 1.6, 11.0, 0.0, "«include»", include_color, '--')


# === Chú thích (Legend) ===
legend_x, legend_y = 14.0, 12.0
ax.plot([legend_x, legend_x + 0.8], [legend_y, legend_y], color=line_color, linewidth=1.5)
ax.text(legend_x + 1.0, legend_y, "Liên kết (association)", fontsize=8.5, va='center', fontfamily='serif')

ax.annotate('', xy=(legend_x + 0.8, legend_y - 0.5), xytext=(legend_x, legend_y - 0.5),
            arrowprops=dict(arrowstyle='->', color=extend_color, lw=1.2, linestyle='--'))
ax.text(legend_x + 1.0, legend_y - 0.5, "«extends»", fontsize=8.5, va='center',
        fontstyle='italic', color=extend_color, fontfamily='serif')

ax.annotate('', xy=(legend_x + 0.8, legend_y - 1.0), xytext=(legend_x, legend_y - 1.0),
            arrowprops=dict(arrowstyle='->', color=include_color, lw=1.2, linestyle='--'))
ax.text(legend_x + 1.0, legend_y - 1.0, "«include»", fontsize=8.5, va='center',
        fontstyle='italic', color=include_color, fontfamily='serif')


# === Tiêu đề ===
fig.text(0.5, 0.98, "Hình 4.1: Sơ đồ Use Case – Ứng dụng Web Phát hiện Deepfake",
         ha='center', va='top', fontsize=15, fontweight='bold',
         fontfamily='serif', color='#1a202c')

plt.tight_layout()
plt.subplots_adjust(top=0.955)

out_dir = os.path.join(os.path.dirname(__file__), '..', 'evaluation_results', 'report_charts')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'hinh_4_1_use_case_diagram.png')
fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white', pad_inches=0.2)
plt.close()
print(f"Saved: {os.path.abspath(out_path)}")
