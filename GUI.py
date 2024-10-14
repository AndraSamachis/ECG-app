from tkinter import *
from customtkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog as fd
from peakdetection import process_and_save_ecg
from main import TreeProcessing

SIZE = '1600x800' #dim fereastra principala
#dim imagini afisate in cadrul aplicatiei
xSize = 1280
ySize = 720

#lista in care stocam caile
image_files = []

#cream instanta principala a ferestrei
root = CTk()
root.geometry(SIZE) #setam dim ferestrei principale
root.configure(bg_color='black') #culoarea de fundal

file_path = ""
def get_back(close_this):
    root.deiconify() #aduce fereastra principala in prim plan
    close_this.destroy() #inchidem fereastra curenta

def get_file_path():
    global file_path
    file_path = fd.askopenfilename() #deschidere dialog de selectie

def button1_command():
    window1 = CTkToplevel(root) #o noua fereastra independenta de root
    window1.geometry(SIZE)
    window1.title("Detecție vârfuri Q, R, S")
    window1.configure(bg_color='black')

    root.withdraw() #ascundem root

    select_file = CTkButton(window1, text="Alege un fișier", command=get_file_path)
    select_file.pack(pady=5, fill=BOTH, expand=True)

    process = CTkButton(window1, text="Procesează")
    process.pack(pady=5, fill=BOTH, expand=True)

    frame = CTkFrame(window1)
    frame.pack(fill=BOTH, expand=True)

    canvas = CTkCanvas(frame, width=1280, height=720, bg='black')
    canvas.pack(side=LEFT, fill=BOTH, expand=True)

    scrollbar = CTkScrollbar(frame, orientation=VERTICAL, command=canvas.yview)
    scrollbar.pack(side=RIGHT, fill=Y)

    canvas.configure(yscrollcommand=scrollbar.set)

    back_btn = CTkButton(window1, text="Înapoi", command=lambda: get_back(window1), fg_color="transparent")
    back_btn.pack(pady=5, fill=BOTH, expand=True)

    global image_files
    global xSize, ySize

    images = []

    def process_plots():
        global file_path
        if file_path != '':
            global image_files
            image_files = process_and_save_ecg(file_path) #stocheaza o lista de cai ale img
            file_path=''
            if (len(image_files) > 0):
                for index, file in enumerate(image_files):
                    img = ImageTk.PhotoImage(Image.open(file).resize((xSize, ySize)))
                    images.append(img)

                    x_position = (canvas.winfo_width() - xSize) // 2 #centarre img
                    y_position = index * ySize #indice
                    canvas.create_image(x_position, y_position, image=img, anchor=NW)


            canvas.config(scrollregion=canvas.bbox("all"))
            window1.images = images



    process.configure(command=process_plots)
    #bucla principala a ferestrei window
    window1.mainloop()


def button2_command():
    window2 = CTkToplevel(root)
    window2.title("Clasifică înregistrări ECG")
    window2.geometry(SIZE)
    window2.configure(bg_color='black')

    root.withdraw()

    select_file = CTkButton(window2, text="Alege un fișier", command=get_file_path)
    select_file.pack(fill=BOTH, pady=5, padx=20)

    choice = StringVar(value="Alege algoritmul")
    no_trees = StringVar()

    #textbox pt a introduce nrul de arbori
    temp_entry = CTkEntry(window2, textvariable=no_trees)  # trebe setata variabila
    temp_button = CTkButton(window2, text="Enter")

    accuracy = StringVar() #obiect ce stocheaza valori de tip string

    def get_accuracy(n, random_forest):
        accuracy.set(TreeProcessing(random_forest, n, False))
        print(TreeProcessing(random_forest, n, False))
        label.configure(text=accuracy.get())

    def manage_choice(option):

        if option == 'Random Forest':
            label.forget()
            back_btn.forget()
            temp_entry.pack()
            temp_button.pack()

        elif option == 'ID3':
            get_accuracy(1, False)

        elif option == 'Alege algoritmul':
            intern_command()


    def intern_command():
        print(no_trees.get())
        temp_button.forget()
        temp_entry.forget()
        label.pack(fill=BOTH, expand=True, pady=20)
        back_btn.pack(pady=5, fill=BOTH)
        if no_trees.get() != "":
            if int(no_trees.get()) > 0:
                get_accuracy(int(no_trees.get()), True)

    temp_button.configure(command=intern_command)

    drop_down_menu = CTkOptionMenu(window2, values=['Alege algoritmul', 'Random Forest', 'ID3'],
                                   variable=choice, command=manage_choice, anchor=CENTER)
    drop_down_menu.pack(fill=BOTH, pady=5, padx=20)

    label = CTkLabel(window2, text="Așteaptă rezultatul...", font=("Helvetica", 40))
    label.pack(fill=BOTH, expand=True, pady=20)

    back_btn = CTkButton(window2, text="Înapoi", command=lambda: get_back(window2))
    back_btn.pack(pady=5, fill=BOTH)


    window2.mainloop()


def main_GUI():
    img = ImageTk.PhotoImage(Image.open("logo.png"))
    lb1 = Label(root, image=img, bg='black')
    lb1.pack()

    b1 = CTkButton(root, text="Detecție vârfuri Q, R, S", command=button1_command)
    b1.pack(fill=BOTH, expand=True, pady=5, padx=5)

    b2 = CTkButton(root, text="Clasifică înregistrări ECG", command=button2_command)
    b2.pack(fill=BOTH, expand=True, pady=5, padx=5)

    exit_btn = CTkButton(root, text="IEȘIRE", command=lambda: root.destroy())
    exit_btn.pack(fill=BOTH, expand=True, pady=5, padx=5)

    root.mainloop()


if __name__ == "__main__":
    main_GUI()