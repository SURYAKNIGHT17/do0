import pandas as pd
# data_set = 'SuperStoreOrders.csv'
# data_set = 'Autism Screening.csv'
# data_set = 'Cleaned-Life-Exp.csv'
data_set = 'Software Engineer Salaries.csv'
df = pd.read_csv(data_set)
column = list(df.columns)
# print(column)
data = {}
for col in column:
    column_name = col
    first_10_values = list(df[column_name].head(5))
    data[col] = first_10_values
# print(data)
data = str(data)
query = "analyse the given snippet of datas set "+data_set+" : " + data +".  provide a pyhthon code to diagnose,preprocess, and all the possible analysis suitable for the data set"
# print(query)
import os
import google.generativeai as genai


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk

# genai.configure(api_key="AIzaSyCxGmyEE5jokRG6LKRKxwE0blo_N5XaORA")

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
  # safety_settings = Adjust safety settings
  # See https://ai.google.dev/gemini-api/docs/safety-settings
)

chat_session = model.start_chat(
  history=[
  ]
)

response = chat_session.send_message(query).text
response = chat_session.send_message("correct the code by rewriting the column names correctly as in the snippet of the data set")
response1 = chat_session.send_message("change the data set source of the code to "+ data_set+" and remove the existing dataset").text
response2 = chat_session.send_message("add all the possible and analysis and visualisations to the existing code which are suitable for the data set and give the full code").text
# response3 = chat_session.send_message("modify the code in a way where all the code comes in try block and the ecxcept block stores the exceptinon as error and give the whole code").text
# print(response.text)
# print(response3)
def trim(response):
    code = response[response.find("```python")+9:]
    code = code[:code.find("```")]
    return code

tcode = trim(response2)
# print(code)
# exec(code)
def write_string_to_file(code_string, filename):
    with open(filename, 'w') as file:
        file.write(code_string)

# Example usage
code = tcode

filename = "generated.py"
write_string_to_file(code, filename)

def debug(code,error):
    query = "debug,correct the python code " +code+" . for the error " +error+" . and provide the whole corrected code . snippet of the dataset  "+data
    response = chat_session.send_message(query).text
    # print("=================== 2nd time =========================")
    # print(response)
    code=trim(response)
    print("------------------------------------------------------")
    print(code)
    write_string_to_file(code,filename)

import subprocess
def py_to_str(filename):
    with open(filename, 'r') as file:
        file_contents = file.read()
    return file_contents

# Run the Python file
round = 1
while(1):
    try:
        result = subprocess.run(['python', 'generated.py'], check=True, capture_output=True, text=True)
        print("Output:  ")
        print(" ")
        print(result.stdout)  # Print the standard output of the script
        break
    except subprocess.CalledProcessError as e:
        # print("An error occurred:")
        error = str(e.stderr)
        print(error)
        code = py_to_str('generated.py')
        debug(code,error)
        print("====================================ROUND : " + str(round) + " ======================================")
        round += 1
