import os
import tkinter as tk
from tkinter import ttk, messagebox
from ttkthemes import ThemedTk
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from youtube import extract_video_id, get_comments, siniflandir, calculate_accuracy

# Yorum verileri
def fetch_and_display_comments():
    url = entry.get()
    video_id = extract_video_id(url)
    if not video_id:
        messagebox.showwarning("Warning", "Lütfen geçerli bir YouTube video URL'si girin.")
        return

    global comments_data
    comments_data = get_comments(video_id)
    if not comments_data:
        messagebox.showinfo("Info", "Yorum bulunamadı.")
        return

    for item in tree.get_children():
        tree.delete(item)

    classified_data = []
    for comment in comments_data:
        comment_text = comment['Yorum']
        classification = siniflandir(comment_text)
        tree.insert('', 'end', values=(
            comment['KanalId'],
            comment['Yorum Yazarı'],
            comment['VideoId'],
            comment['Beğeni Sayısı'],
            comment['Yanıt Sayısı'],
            comment['Tarih'],
            comment_text,
            classification
        ))
        comment['Sınıflandırma'] = classification
        classified_data.append(comment)

    df = pd.DataFrame(classified_data)
    file_path = "comments.csv"
    df.to_csv(file_path, index=False)
    messagebox.showinfo("Success", f"Yorumlar sınıflandırıldı ve {file_path} dosyasına kaydedildi.")

    accuracy = calculate_accuracy(classified_data)
    accuracy_text = (f"Pozitif: %{accuracy['Pozitif']:.2f}, "
                     f"Negatif: %{accuracy['Negatif']:.2f}, "
                     f"Nötr: %{accuracy['Nötr']:.2f}")
    accuracy_label.config(text=accuracy_text)

    # Grafik gösterme
    display_accuracy_chart(accuracy)

# Grafik çizimi
def display_accuracy_chart(accuracy):
    fig, ax = plt.subplots(figsize=(5, 4))
    categories = ["Pozitif", "Negatif", "Nötr"]
    values = [accuracy["Pozitif"], accuracy["Negatif"], accuracy["Nötr"]]
    ax.bar(categories, values, color='skyblue')
    ax.set_title('Sınıflandırma Oranları', fontsize=14)
    ax.set_ylabel('Yüzde (%)')
    ax.set_ylim(0, 100)
    for i, v in enumerate(values):
        ax.text(i, v + 1, f"{v:.2f}%", ha='center', fontsize=10)
    chart_canvas = FigureCanvasTkAgg(fig, master=chart_frame)
    chart_canvas.get_tk_widget().grid(row=0, column=0)
    chart_canvas.draw()

# Tkinter Arayüzü
window = ThemedTk(theme="arc")  
window.title("YouTube Yorum Analizi ve Sınıflandırma")
window.geometry("1200x900")

# Uygulamayı kapatma işlevi
def on_closing():
    if messagebox.askokcancel("Quit", "Uygulamadan çıkmak istediğinize emin misiniz?"):
        window.destroy()
        os._exit(0)

# Ana çerçeve
main_frame = ttk.Frame(window)
main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# URL girişi ve düğme
url_frame = ttk.Frame(main_frame)
url_frame.pack(fill=tk.X, pady=5)

label = ttk.Label(url_frame, text="YouTube Video URL:")
label.pack(side=tk.LEFT, padx=5)

entry = ttk.Entry(url_frame, width=60)
entry.pack(side=tk.LEFT, padx=5)

button = ttk.Button(url_frame, text="Yorumları Getir ve Sınıflandır", command=fetch_and_display_comments)
button.pack(side=tk.LEFT, padx=5)

# Yorumlar tablosu
tree_frame = ttk.Frame(main_frame)
tree_frame.pack(fill=tk.BOTH, expand=True)

columns = ("KanalId", "Yorum Yazarı", "VideoId", "Beğeni Sayısı", "Yanıt Sayısı", "Tarih", "Yorum", "Sınıflandırma")
tree = ttk.Treeview(tree_frame, columns=columns, show='headings')


for col in columns:
    tree.heading(col, text=col)
    tree.column(col, minwidth=100, width=120, anchor='center')


vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=tree.xview)
tree.configure(yscroll=vsb.set, xscroll=hsb.set)
vsb.pack(side='right', fill='y')
hsb.pack(side='bottom', fill='x')
tree.pack(fill=tk.BOTH, expand=True)

# Sınıflandırma başarı oranı ve grafik
stats_frame = ttk.Frame(main_frame)
stats_frame.pack(fill=tk.BOTH, pady=10)

accuracy_label = ttk.Label(stats_frame, text="Pozitif: %0.00, Negatif: %0.00, Nötr: %0.00", font=("Arial", 12))
accuracy_label.pack(pady=5)

chart_frame = ttk.Frame(stats_frame)
chart_frame.pack()

# Tablodaki yorumlara tıklama işlevi
def on_comment_click(event):
    selected_item = tree.selection()
    if selected_item:
        item = tree.item(selected_item)
        comment_text = item['values'][6]
        messagebox.showinfo("Yorum Detayı", comment_text)

tree.bind("<Double-1>", on_comment_click)

# Pencereyi kapatma olayı
window.protocol("WM_DELETE_WINDOW", on_closing)

window.mainloop()
