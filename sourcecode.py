from customtkinter import *
import customtkinter
import threading
import tkinter




class MyWindow:
    def __init__(self, masterx, mastery):


        customtkinter.set_appearance_mode("dark")
        customtkinter.set_default_color_theme("dark-blue")

        self.root = customtkinter.CTk()
        self.root.geometry(f"{masterx}x{mastery}")
        self.root.resizable(width=True, height=True)
        self.root.after(201, lambda :self.root.iconbitmap(r"regression.ico"))
        self.root.title("Construction material's linear regression")
        self.PredictThread = None 

        self.MainCan = tkinter.Canvas(self.root, bg="#1a1a1a", highlightthickness=0)
        
        self.frame = customtkinter.CTkFrame(self.MainCan)
        self.features_frame = customtkinter.CTkFrame(self.frame, border_width=5, border_color="#0f4761")
        self.prediction_frame = customtkinter.CTkFrame(self.frame, border_width=5, border_color="#dbd70a")
        self.training_frame = customtkinter.CTkFrame(self.frame, border_width=5, border_color="#529949")
        self.Title = customtkinter.CTkLabel(self.frame, text="Construction material's concrete compressive strength prediction\nusing linear regression", anchor="center") 
        self.featuretitle = customtkinter.CTkLabel(self.features_frame, text="Input Features") 
        self.predictiontitle = customtkinter.CTkLabel(self.prediction_frame, text="Model Prediction\n(in MPa)")
        self.trainingtitle = customtkinter.CTkLabel(self.training_frame, text="Train Model")
        self.cement = customtkinter.CTkLabel(self.features_frame, text="Cement\n(kg in a m³)") 
        self.entry1 = customtkinter.CTkEntry(self.features_frame, state="disabled")
        self.blastfurnaceslag = customtkinter.CTkLabel(self.features_frame, text="Blast Furnace Slag\n(kg in a m³)") 
        self.entry2 = customtkinter.CTkEntry(self.features_frame, state="disabled")
        self.flyash = customtkinter.CTkLabel(self.features_frame, text="Fly Ash\n(kg in a m³)") 
        self.entry3 = customtkinter.CTkEntry(self.features_frame, state="disabled")
        self.water = customtkinter.CTkLabel(self.features_frame, text="Water\n(kg in a m³)") 
        self.entry4 = customtkinter.CTkEntry(self.features_frame, state="disabled")
        self.superplasticiser = customtkinter.CTkLabel(self.features_frame, text="Superplasticiser\n(kg in a m³)") 
        self.entry5 = customtkinter.CTkEntry(self.features_frame, state="disabled")
        self.coarseaggregate = customtkinter.CTkLabel(self.features_frame, text="Coarse Aggregate\n(kg in a m³)") 
        self.entry6 = customtkinter.CTkEntry(self.features_frame, state="disabled")
        self.fineaggregate = customtkinter.CTkLabel(self.features_frame, text="Fine Aggregate\n(kg in a m³)") 
        self.entry7 = customtkinter.CTkEntry(self.features_frame, state="disabled")
        self.age = customtkinter.CTkLabel(self.features_frame, text="Age\n(days)") 
        self.entry8 = customtkinter.CTkEntry(self.features_frame, state="disabled")
        self.predictionbutton = customtkinter. CTkButton(self.features_frame, state="disabled", text="Predict", command=self.thread_predict, fg_color="#0f4761", font=("Arial (Body CS)", 38), border_width=2, border_color="#1a1a1a")
        self.trainingbutton = customtkinter. CTkButton(self.training_frame, state="disabled", text="Train", command=self.thread_train, fg_color="#529949", font=("Arial (Body CS)", 38), border_width=2, border_color="#1a1a1a", hover_color="#366530")
        self.answer = customtkinter.CTkEntry(self.prediction_frame, state="disabled", fg_color="#8e8b06", border_color="#343638")
        self.checkbox = customtkinter.CTkCheckBox(self.frame, text = "Full screen", command=self.fullscreen)
        self.selectfilebutton=customtkinter.CTkButton(self.training_frame,text="Select dataset", command=self.selectdataset, fg_color="#529949", font=("Arial (Body CS)", 38), border_width=2, border_color="#1a1a1a", hover_color="#366530")
        self.pathdataset=customtkinter.CTkEntry(self.training_frame, state="disabled", fg_color="#529949", border_width=2, border_color="#1a1a1a")
        self.Place(1680,780)
        self.MainCan.bind("<Configure>", self.OnResize)
        self.root.mainloop()

    def Place(self, masterx, mastery):
        self.MainCan.place(relx=0.5, rely=0.5, relheight=1, relwidth=1,anchor=CENTER)
        self.frame.place(relx=0.5, rely=0.5, anchor=CENTER,  relwidth=0.97619, relheight=0.9487179)

        self.features_frame.place(relx=0.4, rely=0.75, anchor=tkinter.CENTER, relwidth=(1250/1680), relheight=(350/780))

        self.prediction_frame.place(relx=0.88, rely=0.6, anchor=tkinter.CENTER, relwidth=(350/1680), relheight=(500/780))

        self.training_frame.place(relx=0.4, rely=0.36, anchor=tkinter.CENTER, relwidth=(1250/1680), relheight=(250/780))

        self.Title.place(relx=0.5, rely=0.12, anchor=tkinter.CENTER)
        self.Title.configure(font=("Arial (Body CS)", (28*((masterx+mastery)/1680))))

        self.featuretitle.place(relx=0.5, rely=0.1, anchor=tkinter.CENTER)
        self.featuretitle.configure(font=("Arial (Body CS)", (20*((masterx+mastery)/1680))))

        self.predictiontitle.place(relx=0.5, rely=0.15, anchor=tkinter.CENTER)
        self.predictiontitle.configure(font=("Arial (Body CS)", (20*((masterx+mastery)/1680))))

        self.trainingtitle.place(relx=0.5, rely=0.15, anchor=tkinter.CENTER)
        self.trainingtitle.configure(font=("Arial (Body CS)", (20*((masterx+mastery)/1680))))

        self.cement.place(relx=0.07, rely=0.25, anchor=tkinter.CENTER)
        self.cement.configure(font=("Arial (Body CS)", (15*((masterx+mastery)/1680))))

        self.entry1.place(relx=0.07, rely=0.55, anchor=tkinter.CENTER, relwidth=(150/1680), relheight=(250/780))
        self.entry1.configure(font=("Arial (Body CS)", (15*((masterx+mastery)/1680))))

        self.blastfurnaceslag.place(relx=(0.07+(1*((1-0.14)/7))), rely=0.25, anchor=tkinter.CENTER)
        self.blastfurnaceslag.configure(font=("Arial (Body CS)", (14*((masterx+mastery)/1680))))

        self.entry2.place(relx=(0.07+(1*((1-0.14)/7))), rely=0.55, anchor=tkinter.CENTER, relwidth=(150/1680), relheight=(250/780))
        self.entry2.configure(font=("Arial (Body CS)", (15*((masterx+mastery)/1680))))

        self.flyash.place(relx=(0.07+(2*((1-0.14)/7))), rely=0.25, anchor=tkinter.CENTER)
        self.flyash.configure(font=("Arial (Body CS)", (15*((masterx+mastery)/1680))))
        
        self.entry3.place(relx=(0.07+(2*((1-0.14)/7))), rely=0.55, anchor=tkinter.CENTER, relwidth=(150/1680), relheight=(250/780))
        self.entry3.configure(font=("Arial (Body CS)", (15*((masterx+mastery)/1680))))

        self.water.place(relx=(0.07+(3*((1-0.14)/7))), rely=0.25, anchor=tkinter.CENTER)
        self.water.configure(font=("Arial (Body CS)", (15*((masterx+mastery)/1680))))

        self.entry4.place(relx=(0.07+(3*((1-0.14)/7))), rely=0.55, anchor=tkinter.CENTER, relwidth=(150/1680), relheight=(250/780))
        self.entry4.configure(font=("Arial (Body CS)", (15*((masterx+mastery)/1680))))

        self.superplasticiser.place(relx=(0.07+(4*((1-0.14)/7))), rely=0.25, anchor=tkinter.CENTER)
        self.superplasticiser.configure(font=("Arial (Body CS)", (12*((masterx+mastery)/1680))))

        self.entry5.place(relx=(0.07+(4*((1-0.14)/7))), rely=0.55, anchor=tkinter.CENTER, relwidth=(150/1680), relheight=(250/780))
        self.entry5.configure(font=("Arial (Body CS)", (15*((masterx+mastery)/1680))))

        self.coarseaggregate.place(relx=(0.07+(5*((1-0.14)/7))), rely=0.25, anchor=tkinter.CENTER)
        self.coarseaggregate.configure(font=("Arial (Body CS)", (12*((masterx+mastery)/1680))))

        self.entry6.place(relx=(0.07+(5*((1-0.14)/7))), rely=0.55, anchor=tkinter.CENTER, relwidth=(150/1680), relheight=(250/780))
        self.entry6.configure(font=("Arial (Body CS)", (15*((masterx+mastery)/1680))))

        self.fineaggregate.place(relx=(0.07+(6*((1-0.14)/7))), rely=0.25, anchor=tkinter.CENTER)
        self.fineaggregate.configure(font=("Arial (Body CS)", (12*((masterx+mastery)/1680))))

        self.entry7.place(relx=(0.07+(6*((1-0.14)/7))), rely=0.55, anchor=tkinter.CENTER, relwidth=(150/1680), relheight=(250/780))
        self.entry7.configure(font=("Arial (Body CS)", (15*((masterx+mastery)/1680))))

        self.age.place(relx=(0.07+(7*((1-0.14)/7))), rely=0.25, anchor=tkinter.CENTER)
        self.age.configure(font=("Arial (Body CS)", (15*((masterx+mastery)/1680))))

        self.entry8.place(relx=(0.07+(7*((1-0.14)/7))), rely=0.55, anchor=tkinter.CENTER, relwidth=(150/1680), relheight=(250/780))
        self.entry8.configure(font=("Arial (Body CS)", (15*((masterx+mastery)/1680))))

        self.predictionbutton.place(relx=0.5, rely=0.85, anchor=tkinter.CENTER, relwidth=(500/1680), relheight=(150/780))
        self.predictionbutton.configure(font=("Arial (Body CS)", (18*((masterx+mastery)/1680))), corner_radius=30)

        self.trainingbutton.place(relx=0.85, rely=0.5, anchor=tkinter.CENTER, relwidth=(300/1680), relheight=(350/780))
        self.trainingbutton.configure(font=("Arial (Body CS)", (18*((masterx+mastery)/1680))), corner_radius=30)

        self.selectfilebutton.place(relx=0.15, rely=0.5, anchor=tkinter.CENTER, relwidth=(300/1680), relheight=(350/780))
        self.selectfilebutton.configure(font=("Arial (Body CS)", (14*((masterx+mastery)/1680))), corner_radius=30)

        self.pathdataset.place(relx=0.5, rely=0.5, anchor=tkinter.CENTER, relwidth=(700/1680), relheight=(350/780))
        self.pathdataset.configure(font=("Arial (Body CS)", (15*((masterx+mastery)/1680))))

        self.answer.place(relx=0.5, rely=0.5, anchor=tkinter.CENTER, relwidth=(1000/1680), relheight=(250/780))
        self.answer.configure(font=("Arial (Body CS)", (18*((masterx+mastery)/1680))))

        self.checkbox.place(relx=0.95, rely=0.96, anchor=CENTER)
        
        
    def selectdataset(self):
        global filename
        filename = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select csv file", filetypes=[("dataset files","*.csv"),])
        self.pathdataset.configure(state="normal")
        self.pathdataset.insert(0, filename)
        self.pathdataset.configure(state="disabled")
        self.trainingbutton.configure(state="normal")
        



    def thread_predict(self):
        
        if self.PredictThread == None:
            self.PredictThread = threading.Thread(target=self.predict)
            self.PredictThread.start()
            return

        if self.PredictThread.is_alive() == False:
            self.PredictThread = threading.Thread(target=self.predict)
            self.PredictThread.start()
            return
        
    def thread_train(self):
        
        if self.PredictThread == None:
            self.PredictThread = threading.Thread(target=self.train)
            self.PredictThread.start()
            return

        if self.PredictThread.is_alive() == False:
            self.PredictThread = threading.Thread(target=self.train)
            self.PredictThread.start()
            return
        
    def train(self):
        self.trainingbutton.place(relx=0.85, rely=0.5, anchor=tkinter.CENTER, relwidth=(300/1680), relheight=(350/780))
        self.trainingbutton.configure(text="Training", state="disabled")
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
        from sklearn.preprocessing import StandardScaler, PolynomialFeatures
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import Ridge
        from sklearn.feature_selection import RFECV
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        global r2, rmse, mae
        global y_test, y_pred, best_model

        data = pd.read_csv(filename)
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]

        X_col = ["Age (day)", "Cement (component 1)(kg in a m^3 mixture)", "Blast Furnace Slag (component 2)(kg in a m^3 mixture)", "Fly Ash (component 3)(kg in a m^3 mixture)", "Water  (component 4)(kg in a m^3 mixture)", 
                        "Superplasticizer (component 5)(kg in a m^3 mixture)", "Coarse Aggregate  (component 6)(kg in a m^3 mixture)", "Fine Aggregate (component 7)(kg in a m^3 mixture)"]
        X = data[X_col]
        y = data["Concrete compressive strength(MPa, megapascals) "]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        pipeline = Pipeline([
            ('poly', PolynomialFeatures(include_bias=False)),
            ('scaler', StandardScaler()),
            ('ridge', Ridge())
        ])
        param_grid = {'poly__degree': [2, 3, 4, 5], 'ridge__alpha': [0.01, 0.1, 1, 10, 100]}


        best_score = float('-inf')
        best_params = None
        for degree in param_grid['poly__degree']:
            for alpha in param_grid['ridge__alpha']:
                pipeline.set_params(poly__degree=degree, ridge__alpha=alpha)
                pipeline.fit(X_train, y_train)
                score = pipeline.score(X_test, y_test)
                if score > best_score:
                    best_score = score
                    best_params = {'poly__degree': degree, 'ridge__alpha': alpha}
        pipeline.set_params(**best_params)
        pipeline.fit(X_train, y_train)
        best_model = pipeline


        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        self.trainingbutton.destroy()
        
        self.resultgbutton = customtkinter. CTkButton(self.training_frame, text="Results", command=self.result, fg_color="#529949", font=("Arial (Body CS)", 28), border_width=2, border_color="#1a1a1a", hover_color="#366530")
        self.graphbutton = customtkinter. CTkButton(self.training_frame, text="Graph", command=self.graph, fg_color="#529949", font=("Arial (Body CS)", 28), border_width=2, border_color="#1a1a1a", hover_color="#366530")
        
        self.resultgbutton.place(relx=0.85, rely=0.3, anchor=tkinter.CENTER, relwidth=(300/1680), relheight=(300/780))
        self.graphbutton.place(relx=0.85, rely=0.7, anchor=tkinter.CENTER, relwidth=(300/1680), relheight=(300/780))

        self.entry1.configure(state="normal")
        self.entry2.configure(state="normal")
        self.entry3.configure(state="normal")
        self.entry4.configure(state="normal")
        self.entry5.configure(state="normal")
        self.entry6.configure(state="normal")
        self.entry7.configure(state="normal")
        self.entry8.configure(state="normal")
        self.predictionbutton.configure(state="normal")


    def result(self):
        self.resultgbutton.configure(state="disabled")
        import numpy as np
        new_window = customtkinter.CTkToplevel(self.root, fg_color="#212121")
        new_window.title("Results")
        new_window.geometry("425x275")
        new_window.resizable(width=False, height=False)


        def close():
            new_window.destroy()
            self.resultgbutton.configure(state="normal")

 
        self.titlewindow = customtkinter.CTkLabel(new_window, text=f"Performance indicator metrics", anchor="center", font=("Arial (Body CS)", 20))
        self.titlewindow.pack(pady=10)

        self.r2 = customtkinter.CTkLabel(new_window, text=f"R² Score: {np.mean(r2):.4f}", anchor="center",font=("Arial (Body CS)", 20))
        self.r2.pack(pady=10)
        
        self.rmse = customtkinter.CTkLabel(new_window, text=f"Root Mean Squared Error (RMSE): {np.mean(rmse):.4f}", anchor="center", font=("Arial (Body CS)", 20))
        self.rmse.pack(pady=10)

        self.mae = customtkinter.CTkLabel(new_window, text=f"Mean Absolute Error (MAE): {np.mean(mae):.4f}", anchor="center", font=("Arial (Body CS)", 20))
        self.mae.pack(pady=10)

        new_button = customtkinter.CTkButton(new_window, text="Close Window", command=close)
        new_button.pack(pady=10)
        
        def confirm():
            new_window.destroy()
            self.resultgbutton.configure(state="normal")

        new_window.protocol("WM_DELETE_WINDOW", confirm)




    def graph(self):
        self.graphbutton.configure(state="disabled")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 5))
        plt.scatter(y_test, y_pred, alpha=0.7)
        plt.xlabel('Actual Concrete Compressive Strength')
        plt.ylabel('Predicted Concrete Compressive Strength')
        plt.title('Actual vs Predicted Values')
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
        plt.show()
        self.graphbutton.configure(state="normal")



    def predict(self):
        self.predictionbutton.place(relx=0.5, rely=0.85, anchor=tkinter.CENTER, relwidth=(700/1680), relheight=(150/780))
        self.predictionbutton.configure(text="predicting",state="disabled")
        import pandas as pd
        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.preprocessing import StandardScaler, PolynomialFeatures
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        import numpy as np
        import matplotlib.pyplot as plt


        cement = self.entry1.get()
        blastfurnaceslag = self.entry2.get()
        flyash = self.entry3.get()
        water = self.entry4.get()
        superplasticiser = self.entry5.get()
        coarseaggregate = self.entry6.get()
        fineaggregate = self.entry7.get()
        age = self.entry8.get()


        _error = False

        try:
            self.entry1.configure(fg_color="#ff0000")
            float(cement)
            self.entry1.configure(fg_color="#343638")
            self.predictionbutton.place(relx=0.5, rely=0.85, anchor=tkinter.CENTER, relwidth=(500/1680), relheight=(150/780))
            self.predictionbutton.configure(text="predict",state="normal")
        except:
            _error = True
        try:
            self.entry2.configure(fg_color="#ff0000")
            float(blastfurnaceslag)
            self.entry2.configure(fg_color="#343638")
            self.predictionbutton.place(relx=0.5, rely=0.85, anchor=tkinter.CENTER, relwidth=(500/1680), relheight=(150/780))
            self.predictionbutton.configure(text="predict",state="normal")
        except:
            _error = True
        try:
            self.entry3.configure(fg_color="#ff0000")  
            float(flyash)
            self.entry3.configure(fg_color="#343638")
            self.predictionbutton.place(relx=0.5, rely=0.85, anchor=tkinter.CENTER, relwidth=(500/1680), relheight=(150/780))
            self.predictionbutton.configure(text="predict",state="normal")
        except:
            _error = True
        try:
            self.entry4.configure(fg_color="#ff0000")
            float(water)
            self.entry4.configure(fg_color="#343638")
            self.predictionbutton.place(relx=0.5, rely=0.85, anchor=tkinter.CENTER, relwidth=(500/1680), relheight=(150/780))
            self.predictionbutton.configure(text="predict",state="normal")
        except:
            _error = True
        try:
            self.entry5.configure(fg_color="#ff0000")
            float(superplasticiser)
            self.entry5.configure(fg_color="#343638")
            self.predictionbutton.place(relx=0.5, rely=0.85, anchor=tkinter.CENTER, relwidth=(500/1680), relheight=(150/780))
            self.predictionbutton.configure(text="predict",state="normal")
        except:
            _error = True
        try:
            self.entry6.configure(fg_color="#ff0000")  
            float(coarseaggregate)
            self.entry6.configure(fg_color="#343638")
            self.predictionbutton.place(relx=0.5, rely=0.85, anchor=tkinter.CENTER, relwidth=(500/1680), relheight=(150/780))
            self.predictionbutton.configure(text="predict",state="normal")
        except:
            _error = True
        try:
            self.entry7.configure(fg_color="#ff0000")
            float(fineaggregate)
            self.entry7.configure(fg_color="#343638")
            self.predictionbutton.place(relx=0.5, rely=0.85, anchor=tkinter.CENTER, relwidth=(500/1680), relheight=(150/780))
            self.predictionbutton.configure(text="predict",state="normal")
        except:
            _error = True
        try:
            self.entry8.configure(fg_color="#ff0000")
            float(age)
            self.entry8.configure(fg_color="#343638")
            self.predictionbutton.place(relx=0.5, rely=0.85, anchor=tkinter.CENTER, relwidth=(500/1680), relheight=(150/780))
            self.predictionbutton.configure(text="predict",state="normal")
        except:
            _error = True
        if _error:
            self.predictionbutton.place(relx=0.5, rely=0.85, anchor=tkinter.CENTER, relwidth=(500/1680), relheight=(150/780))
            self.predictionbutton.configure(text="predict",state="normal")
            return
        
        
        input_data = np.array([[age,cement,blastfurnaceslag,flyash,water,superplasticiser,coarseaggregate,fineaggregate]])
        input_data_transformed = best_model.named_steps['scaler'].transform(
            best_model.named_steps['poly'].transform(input_data)
        )
        predicted_value = best_model.named_steps['ridge'].predict(input_data_transformed)[0]
        
        self.answer.configure(state="normal")
        self.answer.delete(0, 'end')
        self.answer.insert(0, predicted_value)
        self.answer.configure(state="disabled")
        self.predictionbutton.place(relx=0.5, rely=0.85, anchor=tkinter.CENTER, relwidth=(500/1680), relheight=(150/780))
        self.predictionbutton.configure(text="predict",state="normal")
        
    def fullscreen(self):
        if self.checkbox.get():
            self.root.attributes("-fullscreen", True)
        else:
            self.root.attributes("-fullscreen", False)

    def OnResize(self, event):
        self.Place(event.width, event.height)



MyWindow(1680, 780)