# 打包到exe 
# 生成build文件夹和dist, 其中build是临时文件可以删去
# pyinstaller --onefile --windowed --noconfirm AutoArk.py 
# 目前pyinstaller打包wexpect有困难，懒得研究了
pyinstaller --onefile --windowed --noconfirm ArkQuery2.py
cp ./dist/*.exe ./
zip_name="ArkQuery-"
# 清理旧文件
rm $zip_name*.zip -f
# 打包
tar cvf $zip_name$(date -I).zip \
res/* \
config.json \
README.md \
*.exe

# 清空压缩包名
unset zip_name
# 清理临时文件
# rm -r ./build
# rm -r ./dist

echo 'Done!'