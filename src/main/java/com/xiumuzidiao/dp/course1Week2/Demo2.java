package com.xiumuzidiao.dp.course1Week2;

import java.nio.file.Paths;
import java.util.Arrays;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;

import io.jhdf.HdfFile;
import io.jhdf.api.Dataset;

public class Demo2 {
   public static Logger logger = LoggerFactory.getLogger("demo2");


    public static void main(String[] args) {
        INDArray[] i =loadTrainAndTestDataSets("C:\\Users\\220\\Desktop\\train_catvnoncat.h5", "C:\\Users\\220\\Desktop\\test_catvnoncat.h5");
        
      
    }






    public static INDArray[] loadTrainAndTestDataSets(String trainPath,String testPath){
        INDArray[] indArrays = new NDArray[4];
        // 加载训练集的fatures
        try (HdfFile hdfFile = new HdfFile(Paths.get(trainPath))) {
            //获取该hdf文件的train_set_x数据集
            Dataset dataset = hdfFile.getDatasetByPath("train_set_x");
            // 获取该数据集的数据，运行时类是个数组，可以通过dataset.getDataType().getJavaType()来看数据类型是什么，也可以dataset.getDimensions()查看数据集的维度
             logger.info( "训练集feature的数据类型"+dataset.getDataType().getJavaType().toString());
             logger.info(  "训练集feature的维度"+Arrays.toString(dataset.getDimensions()));
            Object trainData = dataset.getData();
            INDArray i = Nd4j.create((int[][][][]) trainData);
            indArrays[0] = i;   

            //加载训练集的labels
            Dataset dataset2 = hdfFile.getDatasetByPath("train_set_y");
            logger.info( "训练集label的数据类型"+dataset2.getDataType().getJavaType().toString());
            logger.info(  "训练集label的维度"+Arrays.toString(dataset2.getDimensions()));
            Object trainData2 = dataset2.getData();
           
            INDArray i2 = Nd4j.create((long[]) trainData2);
            indArrays[1] = i2;   
        }
         // 加载测试集的fatures
         try (HdfFile hdfFile = new HdfFile(Paths.get(testPath))) {
            Dataset dataset = hdfFile.getDatasetByPath("test_set_x");
            logger.info( "测试集feature的数据类型"+dataset.getDataType().getJavaType().toString());
            logger.info(  "测试集feature的维度"+Arrays.toString(dataset.getDimensions()));
            Object testData = dataset.getData();
            
            INDArray i = Nd4j.create((int[][][][]) testData);
            indArrays[2] = i;   

            Dataset dataset2 = hdfFile.getDatasetByPath("test_set_y");
            logger.info( "测试集label的数据类型"+dataset2.getDataType().getJavaType().toString());
            logger.info(  "测试集label的维度"+Arrays.toString(dataset2.getDimensions()));
            Object trainData2 = dataset2.getData();
            
            INDArray i2 = Nd4j.create((long[])  trainData2);
            indArrays[3] = i2;   
        }
        return indArrays;
    }
}
