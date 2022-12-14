package com.xiumuzidiao.dp.course1Week2;

import java.nio.file.Paths;
import java.util.Arrays;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;

import io.jhdf.HdfFile;
import io.jhdf.api.Dataset;

public class Demo2 {
   public static Logger logger = LoggerFactory.getLogger("demo2");


    public static void main(String[] args) {

        INDArray[] i =loadTrainAndTestDataSets("C:\\Users\\xxcxzc\\Desktop\\train_catvnoncat.h5",
         "C:\\Users\\xxcxzc\\Desktop\\test_catvnoncat.h5");           

        INDArray  trainSetxOrig = i[0].castTo(DataType.FLOAT); 
        INDArray  trainSetyOrig = i[1].castTo(DataType.FLOAT); 
        INDArray  testSetxOrig = i[2].castTo(DataType.FLOAT); 
        INDArray  testSetyOrig= i[3].castTo(DataType.FLOAT); 
        INDArray  classOrig = i[4];
        
        
        System.out.println("Number of training examples: m_train = "+ trainSetxOrig.shape()[0]);
        System.out.println("Number of testing examples: m_test =" + testSetxOrig.shape()[0]);
        System.out.println("Height/Width of each image: num_px ="+ testSetxOrig.shape()[1]);
        System.out.println("Each image is of size: ("+  testSetxOrig.shape()[1]+","+testSetxOrig.shape()[2]+",3)");
        System.out.println("train_set_x shape: "+Arrays.toString(trainSetxOrig.shape()));
        System.out.println("train_set_y shape: "+Arrays.toString(trainSetyOrig.shape()));
        System.out.println("test_set_x shape: "+Arrays.toString(testSetxOrig.shape()));
        System.out.println("test_set_y shape: "+Arrays.toString(testSetyOrig.shape()));

        INDArray newTrainset =trainSetxOrig.reshape(trainSetxOrig.shape()[0],trainSetxOrig.shape()[1]*trainSetxOrig.shape()[2]*trainSetxOrig.shape()[3]).transpose();

        INDArray newTestset =testSetxOrig.reshape(testSetxOrig.shape()[0],testSetxOrig.shape()[1]*testSetxOrig.shape()[2]*testSetxOrig.shape()[3]).transpose();
        System.out.println("train_set_x_flatten shape"+Arrays.toString(newTrainset.shape()));
        System.out.println("train_set_y shape: "+Arrays.toString(trainSetyOrig.shape()));
        System.out.println("test_set_x_flatten shape:"+Arrays.toString(newTestset.shape()));
        System.out.println("test_set_y shape: "+Arrays.toString(testSetyOrig.shape()));


        INDArray trainSet = newTrainset.div(255).castTo(DataType.FLOAT);
        
        System.out.println();
        INDArray testSet = newTestset.div(255);
        System.out.println(trainSet.dataType());
        INDArray[] wAndb= initWAndB(12288);
        INDArray w =  wAndb[0];

        INDArray b =  wAndb[1];
        //  w.transpose():1*12228  trainSet:12228*209  b:1*1
        optimize(w.transpose(), b.transpose(), trainSet, trainSetyOrig, 10000, 0.005, false);



    }

    public static INDArray[] loadTrainAndTestDataSets(String trainPath,String testPath){
        Nd4j.setDefaultDataTypes(DataType.FLOAT, DataType.FLOAT);;
        INDArray[] indArrays = new NDArray[5];
        // ??????????????????fatures
        try (HdfFile hdfFile = new HdfFile(Paths.get(trainPath))) {
            //?????????hdf?????????train_set_x?????????
            Dataset dataset = hdfFile.getDatasetByPath("train_set_x");
            // ?????????????????????????????????????????????????????????????????????dataset.getDataType().getJavaType()???????????????????????????????????????dataset.getDimensions()????????????????????????
             logger.info( "?????????feature???????????????"+dataset.getDataType().getJavaType().toString());
             logger.info(  "?????????feature?????????"+Arrays.toString(dataset.getDimensions()));
            Object trainData = dataset.getData();
            INDArray i = Nd4j.createFromArray((int[][][][]) trainData);
        
            indArrays[0] = i;   

            //??????????????????labels
            Dataset dataset2 = hdfFile.getDatasetByPath("train_set_y");
            logger.info( "?????????label???????????????"+dataset2.getDataType().getJavaType().toString());
            logger.info(  "?????????label?????????"+Arrays.toString(dataset2.getDimensions()));
            Object trainData2 = dataset2.getData();
           
            INDArray i2 = Nd4j.createFromArray((long[]) trainData2);
            INDArray i2New = i2.reshape(1,209);
            indArrays[1] = i2New;   
        }
         // ??????????????????fatures
         try (HdfFile hdfFile = new HdfFile(Paths.get(testPath))) {
            Dataset dataset = hdfFile.getDatasetByPath("test_set_x");
            logger.info( "?????????feature???????????????"+dataset.getDataType().getJavaType().toString());
            logger.info(  "?????????feature?????????"+Arrays.toString(dataset.getDimensions()));
            Object testData = dataset.getData();
            
            INDArray i = Nd4j.createFromArray((int[][][][]) testData);
            indArrays[2] = i;   
        // ??????????????????labels
            Dataset dataset2 = hdfFile.getDatasetByPath("test_set_y");
            logger.info( "?????????label???????????????"+dataset2.getDataType().getJavaType().toString());
            logger.info(  "?????????label?????????"+Arrays.toString(dataset2.getDimensions()));
            Object trainData2 = dataset2.getData();
            
            INDArray i2 = Nd4j.createFromArray((long[])  trainData2);
            INDArray i2New = i2.reshape(1,50);
            indArrays[3] = i2New;   

        // ?????????????????????
        Dataset dataset3 = hdfFile.getDatasetByPath("list_classes");
        logger.info( "???????????????????????????"+dataset3.getDataType().getJavaType().toString());
        logger.info(  "?????????????????????"+Arrays.toString(dataset3.getDimensions()));
        Object trainData3 = dataset3.getData();
            
        INDArray i3 = Nd4j.create((String[])  trainData3);
        INDArray i3New = i3.reshape(1,2);
        indArrays[4] = i3New;   
        }
        return indArrays;
    }
    public static INDArray sigmoid(INDArray z){
        
        return Nd4j.math.pow(Nd4j.math().exp(z.mul(-1)).add(1),-1.0);
    }
    /**
     * 
     * @param dim ????????????w???b????????????????????????????????? dim*1
     * @return  ??????w???b???w???dim*1???b??????dim*1????????????????????????
     */
    public static INDArray[] initWAndB(int dim){

        INDArray w = Nd4j.zeros(dim,1);
        INDArray b = Nd4j.zeros(1, 1);
        return new INDArray[]{w,b};
    }
    public static Object[] propagate(INDArray w, INDArray b, INDArray x, INDArray y){
        double m = x.shape()[1];
        INDArray a = sigmoid(w.mmul(x).add(b));


        //??????a = sigmoid(w.mmul(x).add(b));
        logger.info("a= sigmoid(wx+b)"+a.toString());
        //cost = -(1.0 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) 


        //??????  np.log(A)
        INDArray tempA = Transforms.log(a);

        logger.info("np.log(A)"+tempA.toString());


        //?????? np.log(A)????????????
        INDArray tempB = tempA.broadcast(3,3);
        logger.info(("np.log(A)????????????"+tempB.toString()));
        logger.info(("??????????????????????????????????????????element-wise ??????"));



        // ??????Y * np.log(A)
        INDArray tempC =  y.mul(tempA);
        logger.info("Y "+y); 
        logger.info("Y * np.log(A)"+tempC);  
        

        INDArray tempD = a.mul(-1).add(1);logger.info("1- A"+tempD);

        INDArray tempE = Transforms.log(tempD);    
        
        logger.info("np.log(1 - A)) " + tempE);


        INDArray tempF = y.mul(-1).add(1).mul(tempE);


        logger.info("(1 - Y) * np.log(1 - A) " + tempF);

        INDArray tempG = tempF.add(tempC);
        logger.info("Y * np.log(A) + (1 - Y) * np.log(1 - A)"+tempG);

        double sum = tempG.sumNumber().doubleValue();
        logger.info(" np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) "+sum);
        double cost = -1.0 / m  * sum; 
        logger.info(" (propagateOne)cost: "+cost);

        INDArray dw = x.mmul(a.sub(y).transpose()).div(m);
        Number db = a.sub(y).sumNumber();
               db = (double)db /m;
        return new Object[]{cost,dw,db};
    }

    // ????????????
    public static Object[] propagateTwo(INDArray w, INDArray b, INDArray x, INDArray y){
        double m = x.shape()[1];
        INDArray a = sigmoid(w.mmul(x).add(b));

        //cost = -(1.0 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) 

        INDArray tempA = (y.mul(Transforms.log(a)))    
        .add(
            ( y.mul(-1).add(1)) .mul(Transforms.log(a.mul(-1).add(1)))
            );    


        double sum = tempA.sumNumber().doubleValue();
        logger.info(" (propagateTwo)np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) "+sum);
        double cost = -1.0 / m  * sum; 
        logger.info("(propagateTwo) cost: "+cost);
        INDArray dw = x.mmul(a.sub(y).transpose()).div(m);
        Number db = a.sub(y).sumNumber();
               db = (double)db /m;
        return new Object[]{cost,dw,db};
    }

    /**
     * sigmoid(w.mmul(x).add(b)) ?????????sigmoid???????????????????????????????????????
     * @param w  ?????????????????????????????????1*n  
     * @param b  ?????????????????????????????????1*n
     * @param x  feture??????????????????n*m???m?????????????????????n???????????????????????????
     * @param y  ?????????labels??????????????????1*m
     * @return ?????????cost(double) ,dw(??????,?????????n*1),db(double)
     * 
     * ????????????????????????labels??????????????????w???b??????????????????????????????????????????????????????
     */
    // ????????????
    public static Object[] propagateThree(INDArray w, INDArray b, INDArray x, INDArray y){
        double m = x.shape()[1];
        INDArray a = sigmoid(w.mmul(x).add(b));
        //cost = -(1.0 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) 

        INDArray tempA = (y.mmul(Transforms.log(a).transpose()))    
        .add(
            ( y.mul(-1).add(1)) .mmul(Transforms.log(a.mul(-1).add(1).transpose()))
            );    

        INDArray resul = tempA.mul(-1.0/m);
        
        double cost = (double) resul.sumNumber();
        
        logger.info(" (propagateThree)cost: "+cost);
        INDArray dw = x.mmul(a.sub(y).transpose()).div(m);
        Number db = a.sub(y).sumNumber();
               db = (double)db /m;
        return new Object[]{cost,dw,db};
    }

    public static void optimize(INDArray w, INDArray b, INDArray x, INDArray y, int numOfIterations , double learningRate , boolean printCost){
          
        for(int i = 0 ;i<numOfIterations;i++){
            //?????????????????????????????????cost???w????????????b?????????
            Object[] objs = propagateThree(w,b,x,y);
            double cost = (double) objs[0];
            INDArray dw =  (INDArray) objs[1];
            Double db =  (Double) objs[2];
            if(i == numOfIterations-1){
                System.out.println("w????????????");
                System.out.println(w);
                System.out.println("b????????????");
                System.out.println(b);
                System.out.println("dw????????????");
                System.out.println(dw);
                System.out.println("db????????????");
                System.out.println(db);

            }
            // ??????w???b w= w-lr*dw
            w  = w.transpose().sub(dw.mul(learningRate)).transpose();
            b  = b.transpose().sub(db*learningRate).transpose();
           
        }



    }   

}
