from zipfile import ZipFile
import os
import glob

file = 'AutoArk.zip'
prefix = './AutoArk/'

try:
    os.remove(file)
except:
    pass
    
# create a ZipFile object
with ZipFile(file, 'w') as zipObj:
    # Add multiple files to the zip
    zipObj.write('./dist/AutoArk.exe', prefix + 'AutoArk.exe')
    
    for i in ['config.json', 'README.md', 'LICENSE']:
        zipObj.write(i, prefix + i)
    
    for i in glob.glob('./img/*'):
        zipObj.write(i, prefix + i)
        
    for i in glob.glob('./res/*'):
        zipObj.write(i, prefix + i)
