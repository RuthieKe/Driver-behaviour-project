import gradio as gr
import pickle
import os
import shap
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import seaborn as sns
import pandas as pd


data = "cleaned_dataset.csv"
df= pd.read_csv(data)

def load_data(data):
    # Reads the CSV file and returns the first few rows
    data = pd.read_csv(data)
    return data.head()


def make_risk_assessment(Overspeeding, OverRPM, Acceleration, Deceleration, Idle_time, Duration, Distance):
    with open("model.pkl", "rb") as f:
        clf  = pickle.load(f)

    features = [[Overspeeding, OverRPM, Acceleration, Deceleration, Idle_time, Duration, Distance]]

    preds = clf.predict(features)

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(features)

    shap.initjs()
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, features, feature_names=["Overspeeding", "OverRPM", "Acceleration", "Deceleration", "Idle_time", "Duration", "Distance"], show=False)
    
    # Save plot to a BytesIO buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)

    #if preds[0] == 1:
    #        return "Your Driving has been Assessed as Safe Driving"
    #return "Your Driving has been Assessed as Risky Driving" 

    # Generate the risk assessment message
    risk_assessment_message = "Your Driving has been Assessed as Safe Driving" if preds[0] == 1 else "Your Driving has been Assessed as Risky Driving"

    # Return both the message and the image buffer
    return risk_assessment_message, img

# Univariate Analysis
def univariate_analysis(feature):
    plt.figure(figsize=(8, 4))
    sns.histplot(df[feature], kde=True)
    plt.title(f'Histogram of {feature}')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return Image.open(buf)

# Bivariate Analysis
def bivariate_analysis(feature1, feature2):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=feature1, y=feature2)
    plt.title(f'Scatter Plot between {feature1} and {feature2}')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return Image.open(buf)

# Multivariate Analysis
def multivariate_analysis():
    plt.figure(figsize=(10, 8))
    sns.pairplot(df)
    plt.title('Pairplot of all features')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return Image.open(buf)

#css = ".gradio-container {background: url(https://stock-vip-processed-prod-irl1.s3.eu-west-1.amazonaws.com/5886a6dab5.ai?AWSAccessKeyId=AKIAUMGGMQGESZGA4YHJ&Expires=1714298845&Signature=qgq%2FJx3RsKJx8hwUpLbIJ6QB%2FMM%3D&response-content-disposition=attachment%3B%20filename%3D%22AdobeStock_243118595_Preview.ai%22&response-content-type=application%2Fillustrator)}"

image_path = '/Users/wkiruki/Downloads/photo-1528460033278-a6ba57020470.avif' # Replace with your image file path

absolute_path = os.path.abspath(image_path)

                         
#name_input = gr.Textbox(label = "Enter Driver's Name: ")
#Id_input = gr.Textbox(label = "Enter Driver's ID: ")                           
overspeeding_input = gr.Slider(label = "Enter the Overspeeding Score of the Driver:")
overRPM_input = gr.Slider(label= "Enter the Over RPM (Revolutions Per Minute) Score of the Driver:")
acceleration_input = gr.Slider(label = "Enter the Acceleration Score of the Driver:")
deceleration_input = gr.Slider(label = "Enter the Deceleration Score of the Driver:")
idle_time_input = gr.Slider(label = "Enter the Idle_time Score of the Driver:")
distance_input = gr.Number(label = "Enter the Distance Covered by the Driver:")
duration_input = gr.Number(label = "Enter the Duration Taken by the Driver:")

univariate_input = gr.Dropdown(list(df.columns[:-1]), label="Select Feature for Univariate"),
bivariate1_input = gr.Dropdown(list(df.columns[:-1]), label="Feature 1 for Bivariate"),
bivariate2_input = gr.Dropdown(list(df.columns[:-1]), label="Feature 2 for Bivariate"),



# We create the output
output = gr.Textbox(label="Risk Assessment")
shap_output = gr.Image(label="SHAP Impact Plot")

analysis_output = [gr.Image() for _ in range(3)]



model_prediction_interface = gr.Interface(fn = make_risk_assessment, 
                   inputs=[overspeeding_input, overRPM_input, acceleration_input, deceleration_input, idle_time_input, distance_input, duration_input], 
                   outputs=[output,shap_output],
                   title= "RISK ASSESSMENT FOR FLEET MANAGED DRIVERS",
                   css= ".gradio-container {background: url('file=/Users/wkiruki/Downloads/photo-1528460033278-a6ba57020470.avif')}")

univariate_interface = gr.Interface(
    fn=univariate_analysis,
    inputs=[gr.Dropdown(list(df.columns[:-1]), label="Select Feature for Univariate")],
    outputs=[gr.Image()],
    title="Univariate Analysis"
)

bivariate_interface = gr.Interface(
    fn=bivariate_analysis,
    inputs=[
        gr.Dropdown(list(df.columns[:-1]), label="Feature 1 for Bivariate"),
        gr.Dropdown(list(df.columns[:-1]), label="Feature 2 for Bivariate")
    ],
    outputs=[gr.Image()],
    title="Bivariate Analysis"
)

multivariate_interface = gr.Interface(
    fn=multivariate_analysis,
    inputs=[],
    outputs=[gr.Image()],
    title="Multivariate Analysis"
)
 
csv_interface = gr.Interface(
    fn=load_data,  # Function to call when a file is uploaded
    inputs=[gr.File(label="Upload your CSV file")],  # File input for uploading CSV
    outputs=[gr.Dataframe()],  # Output the results as a DataFrame
    title="CSV Data Loader"  # Title of the interface
)


# Combine into a tabbed interface
app = gr.TabbedInterface(
    [csv_interface, univariate_interface, bivariate_interface, multivariate_interface, model_prediction_interface],
    ["LOAD DATA", "UNIVARIATE", "BIVARIATE", "MULTIVARIATE", "RISK ASSESSMENT FOR FLEET MANAGED DRIVERS"]
)


app.launch(allowed_paths=[absolute_path])