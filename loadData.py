import os
import cv2
import pandas as pd

images_list, texts_list =  [], []

def RGB_values(file_path_image):
    image = cv2.imread(file_path_image)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb
    
def readTextFile(file_path_text):
    with open(file_path_text, errors = 'ignore') as f:  # read the text in the text file.
        lines = f.read()
        return lines
    
# Sample Load Images. Assuming all files of images and text are in a folder
def load_images(file_path_image):
    for filename in os.listdir(file_path_image):
         if (filename.lower().endswith(('.jpg', '.png', '.jpeg'))):
             image_file_path = os.path.join(file_path_image, filename)
             rgb_values_image = RGB_values(image_file_path) # appending the RGB values to the text
             number = filename.split('.')[0]  # get the number in the filename for the index
             images_list.append({"Index": number, "RGB-values": rgb_values_image})
             
    return images_list

# Load Text

def load_text_data(file_path_text):
    for fileNameText in os.listdir(file_path_text):
         if (fileNameText.lower().endswith('.txt') and not (fileNameText.lower().endswith("all.txt"))):
             text_file_path = os.path.join(file_path_text, fileNameText)
             text_for_data = readTextFile(text_file_path)
             number_text = fileNameText.split('.')[0] # retrives the index
             texts_list.append({"Index": number_text, "Text": text_for_data})
             
    return texts_list


def load_labels(file_path):
    for filename in os.listdir(file_path):
        if (filename.lower().endswith('all.txt')):
            labels_path = os.path.join(file_path, filename)
            data3 = pd.read_csv(labels_path, delimiter = '\t') # read the csv file
            data3[['Text', 'Image']] = data3['text,image'].str.split(',', expand=True)  # expand = True to separate the arrays into distinct columns
        
            data3.drop(columns=['text,image'], inplace=True)
        
            data3.columns = ['Index', 'TextLabel', 'ImageLabel']
        
            merged_df2 = pd.merge(combineImageText(images_list, texts_list), data3, on = "Index")
            
            merged_df2.reindex(['Index, RGB-values, ImageLabel, Text, TextLabel'], axis = 1)
        
            return merged_df2
        
        
def combineImageText(imageList, textList):
    dataFrame = pd.DataFrame(imageList)
    dataFrame2 = pd.DataFrame(textList)
    
    dataFrame.Index = dataFrame.Index.astype(int)
    dataFrame2.Index = dataFrame2.Index.astype(int)

    dataFrame = dataFrame.sort_values("Index", ascending = True)  # arrange the columns in ascending order.
    dataFrame2 = dataFrame2.sort_values("Index", ascending = True)

    merged_df = pd.merge(dataFrame, dataFrame2, on = "Index") # merge dataframe on the Index value
    
    return merged_df
