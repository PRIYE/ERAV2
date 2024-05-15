## YoloV3

### YoloV3 Simplified for training on Colab with custom dataset.
1. Clone this repo: https://github.com/miki998/YoloV3_Annotation_Tool
2. Follow the installation steps as mentioned in the repo.
3. For the assignment, download 500 images of your unique object.
4. Annotate the images using the Annotation tool.

As per the instruction above, I have added the class 'SafetyVest' as I am trying to detect Safety Vest in the images of construction workers. The concerned folder in data is marked 'customdata'.

### Result

1. Correct detection (True Positive) 
   ![road_construction_asphalt_work_machinery_street_site_workers-743811_jpg rf 75b24aea212c272f24747b3f729a5517](https://github.com/PRIYE/ERAV2/assets/7592375/25f0df1b-3719-4e35-ad94-bf81c69c348c)
   ![ppe_1292_jpg rf f27de1f34080518e60b406c5c9dd8e91](https://github.com/PRIYE/ERAV2/assets/7592375/c288cc32-3702-47bd-b4d7-a7ae528d7ebf)
   ![image_27_jpg rf 3a4f2c0f85b91bf818b2a3dda01692d9](https://github.com/PRIYE/ERAV2/assets/7592375/48990ce3-c96d-4efe-b2d6-cb19044f4e05)


3. Missed detection
   ![ppe_1327_jpg rf b8d5c1f1cd2deec5728e5562c3a989d9](https://github.com/PRIYE/ERAV2/assets/7592375/abbc4c8c-52d4-4a77-83a2-62093fde4be5)
   ![ppe_1325_jpg rf a8d648eccda6a0b8bd71822dc5f5d6dc](https://github.com/PRIYE/ERAV2/assets/7592375/fdc85a51-953b-4b0d-b510-c42d4e09c066)
   ![ppe_1302_jpg rf c930f9e3e03ac9c4892cfba7d41ac63a](https://github.com/PRIYE/ERAV2/assets/7592375/c156bf38-310f-4314-8478-1e6fdf1d27ec)

4. Partial Detection
   ![image](https://github.com/PRIYE/ERAV2/assets/7592375/b3597d44-afe6-4835-82b6-1b2d7abbb61d)
   ![image](https://github.com/PRIYE/ERAV2/assets/7592375/202c5b48-8037-410a-807c-a678148b8035)

5. Correct detection (True Negative)
   ![ppe_1328_jpg rf 617faaad15175f0a7eab0002aebb7bb3](https://github.com/PRIYE/ERAV2/assets/7592375/cf5b9f1c-49e9-4a2d-83ca-71eca157f169)
   ![ppe_1321_jpg rf 39c5254b5bc698c579d785c2df811d4f](https://github.com/PRIYE/ERAV2/assets/7592375/e6758f90-caf7-4c04-b3fa-32b5269c08cc)

### Training Logs
![image](https://github.com/PRIYE/ERAV2/assets/7592375/d3613850-bf1f-4f2b-a6c3-114ca5886549)
![image](https://github.com/PRIYE/ERAV2/assets/7592375/4df10905-80a2-4181-b846-4decdc3a2d70)
![image](https://github.com/PRIYE/ERAV2/assets/7592375/1c2c7c6a-0161-43f7-a033-343201332e46)
![image](https://github.com/PRIYE/ERAV2/assets/7592375/9c38422a-e4e7-4639-a038-0caa92439aaa)

### Performance Metric
![image](https://github.com/PRIYE/ERAV2/assets/7592375/56bfe648-1857-47d7-a644-9a56dc06ba1f)
