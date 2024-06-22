import gdown

url = "https://drive.google.com/file/d/1QZIGzI7-f4AD1-r022QJ80E1Wo9vXoNr/view?usp=sharing"
output = 'control_probs.zip'
gdown.download(url, output, fuzzy=True)