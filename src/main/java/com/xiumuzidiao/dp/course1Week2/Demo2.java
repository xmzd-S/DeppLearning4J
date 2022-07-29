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

        INDArray[] i =loadTrainAndTestDataSets("C:\\Users\\220\\Desktop\\train_catvnoncat.h5",
         "C:\\Users\\220\\Desktop\\test_catvnoncat.h5");           

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
        // 加载训练集的fatures
        try (HdfFile hdfFile = new HdfFile(Paths.get(trainPath))) {
            //获取该hdf文件的train_set_x数据集
            Dataset dataset = hdfFile.getDatasetByPath("train_set_x");
            // 获取该数据集的数据，运行时类是个数组，可以通过dataset.getDataType().getJavaType()来看数据类型是什么，也可以dataset.getDimensions()查看数据集的维度
             logger.info( "训练集feature的数据类型"+dataset.getDataType().getJavaType().toString());
             logger.info(  "训练集feature的维度"+Arrays.toString(dataset.getDimensions()));
            Object trainData = dataset.getData();
            INDArray i = Nd4j.createFromArray((int[][][][]) trainData);
        
            indArrays[0] = i;   

            //加载训练集的labels
            Dataset dataset2 = hdfFile.getDatasetByPath("train_set_y");
            logger.info( "训练集label的数据类型"+dataset2.getDataType().getJavaType().toString());
            logger.info(  "训练集label的维度"+Arrays.toString(dataset2.getDimensions()));
            Object trainData2 = dataset2.getData();
           
            INDArray i2 = Nd4j.createFromArray((long[]) trainData2);
            INDArray i2New = i2.reshape(1,209);
            indArrays[1] = i2New;   
        }
         // 加载测试集的fatures
         try (HdfFile hdfFile = new HdfFile(Paths.get(testPath))) {
            Dataset dataset = hdfFile.getDatasetByPath("test_set_x");
            logger.info( "测试集feature的数据类型"+dataset.getDataType().getJavaType().toString());
            logger.info(  "测试集feature的维度"+Arrays.toString(dataset.getDimensions()));
            Object testData = dataset.getData();
            
            INDArray i = Nd4j.createFromArray((int[][][][]) testData);
            indArrays[2] = i;   
        // 加载测试集的labels
            Dataset dataset2 = hdfFile.getDatasetByPath("test_set_y");
            logger.info( "测试集label的数据类型"+dataset2.getDataType().getJavaType().toString());
            logger.info(  "测试集label的维度"+Arrays.toString(dataset2.getDimensions()));
            Object trainData2 = dataset2.getData();
            
            INDArray i2 = Nd4j.createFromArray((long[])  trainData2);
            INDArray i2New = i2.reshape(1,50);
            indArrays[3] = i2New;   

        // 加载图片的类型
        Dataset dataset3 = hdfFile.getDatasetByPath("list_classes");
        logger.info( "图片类型的数据类型"+dataset3.getDataType().getJavaType().toString());
        logger.info(  "图片类型的维度"+Arrays.toString(dataset3.getDimensions()));
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
     * @param dim 要得到的w和b的维度，它们的维度都是 dim*1
     * @return  返回w和b，w是dim*1，b也是dim*1，它们都是零向量
     */
    public static INDArray[] initWAndB(int dim){

        INDArray w = Nd4j.zeros(dim,1);
        INDArray b = Nd4j.zeros(1, 1);
        return new INDArray[]{w,b};
    }
    public static Object[] propagate(INDArray w, INDArray b, INDArray x, INDArray y){
        double m = x.shape()[1];
        INDArray a = sigmoid(w.mmul(x).add(b));


        //打印a = sigmoid(w.mmul(x).add(b));
        logger.info("a= sigmoid(wx+b)"+a.toString());
        //cost = -(1.0 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) 


        //打印  np.log(A)
        INDArray tempA = Transforms.log(a);

        logger.info("np.log(A)"+tempA.toString());


        //打印 np.log(A)广播之后
        INDArray tempB = tempA.broadcast(3,3);
        logger.info(("np.log(A)广播之后"+tempB.toString()));
        logger.info(("其实这里并没有用到广播，而是element-wise 乘积"));



        // 打印Y * np.log(A)
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

    // 简化版本
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
     * sigmoid(w.mmul(x).add(b)) 然后用sigmoid函数作用，随后代入损失函数
     * @param w  权重矩阵的转置，维度是1*n  
     * @param b  偏差矩阵的转置，维度是1*n
     * @param x  feture矩阵，维度是n*m，m是样本的数量，n是每个样本的特征数
     * @param y  实际的labels矩阵，维度是1*m
     * @return 返回了cost(double) ,dw(矩阵,维度是n*1),db(double)
     * 
     * 样本按照列堆叠，labels也按列堆叠，w和b也都按列堆叠，但是传入的是它们的转置
     */
    // 简化版本
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
            //前向传播及后向传播得到cost，w的梯度和b的梯度
            Object[] objs = propagateThree(w,b,x,y);
            double cost = (double) objs[0];
            INDArray dw =  (INDArray) objs[1];
            Double db =  (Double) objs[2];
            if(i == numOfIterations-1){
                System.out.println("w矩阵为：");
                System.out.println(w);
                System.out.println("b矩阵为：");
                System.out.println(b);
                System.out.println("dw矩阵为：");
                System.out.println(dw);
                System.out.println("db矩阵为：");
                System.out.println(db);

            }
            // 更新w和b w= w-lr*dw
            w  = w.transpose().sub(dw.mul(learningRate)).transpose();
            b  = b.transpose().sub(db*learningRate).transpose();
           
        }



    }   

}
