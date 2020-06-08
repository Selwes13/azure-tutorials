# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import pickle
from datetime import datetime
from azureml.core import Workspace, Datastore, Dataset, Model

#%% connect to workspace
SUBSCRIPTION_ID = “########–####–####-####–########” 
RESOURCE_GROUP = “resource-group-name” 
WORKSPACE_NAME = “workspace-name”

ws = Workspace(SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME)

print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep = '\n')

#%% List all blobstores
for store in ws.datastores:
    print(store)
    
#%% connect to a datastore by name
blob_store = Datastore(ws, "workspaceblobstore")
print("\n" + f"Connected to blobstore: {blob_store.name}")

#%% 
# ** Downloading training data from the Blobstore **
#%% download the file from the blobstore to your local machine (or compute instance)
blob_path = "Campus_Recruitment/Raw_Data"# if None will download whole blob
local_path = "./"

blob_store.download(target_path=local_path,
                    prefix=blob_path, 
                    overwrite=True, 
                    show_progress=True)

#%% Load the data into memory
df = pd.read_csv("./Campus_Recruitment/Raw_Data/datasets_596958_1073629_Placement_Data_Full_Class.csv")
df[:10]

#%% 
# ** Transform the data for use with our model **
#%% transform some column values into numeric and categorical
df['male'] = df['gender'].map({'M':1, 'F':0})
df['education_board_is_central'] = df['ssc_b'].map({'Central':1, 'Others':0})
df['hsc_s'] = df['hsc_s'].map({'Commerce':0 , 'Science':1 , 'Arts':2})
df['workex_bool'] = df['workex'].map({'Yes':1, 'No':0})

#%% drops nan values in columns we are using for the pca
training_columns = [
                    'male', 'ssc_p', 'education_board_is_central', 
                    'hsc_p', 'degree_p', 'workex_bool', 
                    'etest_p', 'mba_p', 'salary'
                   ]

df_pca = df.dropna(subset=training_columns)

print(f"{len(df)} -> {len(df_pca)}")

#%% 
# ** Train the PCA Model **
#%% train the PCA model using these new columns

# test different numbers of components
for c in range(1, len(training_columns)):
    pca_model = PCA(n_components=c)
    pca_model = pca_model.fit(df_pca[training_columns])
    
    # calculates how much of the variance is captured by the PCA
    exp_var = round(np.sum(pca_model.explained_variance_ratio_) * 100, 2)
    print(f"{c} components -> {exp_var}% of variance captured")
    
#%% I am going to use 4 components as it captures 95% of the variance while halving the number of columns
# train the new PCA Model
pca_model = PCA(n_components=4)
pca_model = pca_model.fit(df_pca[training_columns])
    
#%%
# ** Upload the data to the Blobstore **
#%% first save the files to disk
if (not os.path.exists("./Upload")):
    os.mkdir("./Upload")
    os.mkdir("./Upload/Data")
    os.mkdir("./Upload/Model")
    
df_pca.to_csv("./Upload/Data/data.csv", index=False)
pickle.dump( pca_model, open( "./Upload/Model/model.pkl", "wb" ) )

#%% now you can upload that directory to blobstorage
# I use the date to diferentiate the different versions
blob_path = f"Campus_Recruitment/{datetime.now().strftime('%Y-%m-%d')}"# if None will upload to root
local_path = "./Upload/Data"

blob_store.upload(src_dir=local_path, 
                  target_path=blob_path,
                  overwrite=True, 
                  show_progress=True)

#%% 
# ** Register the data as a dataset **
# %% now that the data is up on the blobstore we can register it as a dataset 
# to keep track of its versions and make it easily acessible
dataset = Dataset.File.from_files( blob_store.path(blob_path + "/data.csv") )
dataset.register(ws, 
                 name="Campus_Recruitment_PCA_Training_Data",
                 create_new_version=True)

#%% 
# ** Upload and register the model as a Model **
#%% 
model = Model.register(workspace=ws,
                       model_name='Campus_Recruitment_PCA',                # Name of the registered model in your workspace.
                       model_path='./Upload/Model/model.pkl',  # Local file to upload and register as a model.
                      
                       sample_input_dataset=dataset,
                       sample_output_dataset=None,
                      
                       description='PCA model for dimention reduction of the Campus Recruitment Dataset',
                      )

print('Name:', model.name)
print('Version:', model.version)

