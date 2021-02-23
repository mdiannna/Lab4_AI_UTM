import urllib.request
import json
import os
import ssl
from termcolor import colored

data = {"data": [{"apartmentsNr": "126.000000",
        "col1": "-122.230000",
        "col2": "37.880000",
        "col8": "8.325200",
        "complexAge": "41.000000",
        "complexInhabitants": "322.000000",
        "totalBedrooms": "129.000000",
        "totalRooms": "880.000000"}]}

COLUMN_NAMES = ["col1", "col2", "complexAge", "totalRooms", "totalBedrooms", "complexInhabitants", "apartmentsNr", "col8", "medianCompexValue"]

if  __name__=="__main__":
    s_in = input("Choose an option to load data: [from_file or from_input]\n>>")

    if s_in.lower() not in ["from_file", "from_input"]:
        print(colored("Wrong option. Please try again. Quitting.", "red"))
        quit()

    if s_in.lower()=="from_file":
        json_filename = ""
        while json_filename[-5:]!=".json":

            json_filename = input("Please write path to a json file:\n>>")
            if json_filename[-5:]!=".json":
                print(colored("Wrong file format. Please choose json file", "red"))

        with open(json_filename) as f:
            data = json.load(f)

    elif s_in.lower()=="from_input":
        for col in COLUMN_NAMES:
            if col!="medianCompexValue":
                col_data = input("Please write value for column '" + col + " '\n"+ col +">>")
                try:
                    col_data = float(col_data)
                except:
                    print(colored("Wrong data format. Must be float. Quiting"), "red")
                    quit()



def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get("PYTHONHTTPSVERIFY", "") and getattr(ssl, "_create_unverified_context", None):
        ssl._create_default_https_context = ssl._create_unverified_context

allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.



body = str.encode(json.dumps(data))

url = "http://9a898eb5-4f48-4f92-b645-23926138a1d8.westeurope.azurecontainer.io/score"
api_key = "" # Replace this with the API key for the web service
headers = {"Content-Type":"application/json", "Authorization":("Bearer "+ api_key)}

req = urllib.request.Request(url, body, headers)

try:
    response = urllib.request.urlopen(req)

    result = response.read()
    print("Prediction result(s):", result)
except urllib.error.HTTPError as error:
    print("The request failed with status code: " + str(error.code))

    # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
    print(error.info())
    print(json.loads(error.read().decode("utf8", "ignore")))
