package com.xiumuzidiao.dp.course1Week2;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Demo1 {
    public static void main(String[] args) {
        new Demo1().test1();
    }

    //Exercise 1  Build a function that returns the sigmoid of a real number x. Use math.exp(x) for the exponential function.
    @Test
    public void test1() {
        System.out.println(sigmoidOne(3));

    }
    // Exercise2 Implement the sigmoid function using numpy.
    @Test
    public void test2(){
        System.out.println(sigmoidTwo(Nd4j.createFromArray(1.0, 2.0, 3.0)));

    }

    //  Compute the gradient (also called the slope or derivative) of the sigmoid function with respect to its input x
    @Test
    public void test3(){

        System.out.println(sigmoidGrad(Nd4j.createFromArray(1.0, 2.0, 3.0)));



    }

    //Implement `image2vector()` that takes an input of shape (length, height, 3) and returns a vector of shape (length\*height\*3, 1). For example, if you would like to reshape an array v of shape (a, b, c)
    // into a vector of shape (a*b,c) you would do:

    @Test
    public void test4(){
        INDArray rand = Nd4j.rand(2, 3, 3);
        System.out.println(image2vector(rand));
    }
    //Implement normalizeRows() to normalize the rows of a matrix. After applying this function to an input matrix x, each row of x should be a vector of unit length (meaning length 1).
    @Test
    public void test5(){
        INDArray rand = Nd4j.createFromArray(0.0,3.0,4.0,1.0,6.0,4.0 );
        INDArray reshape = rand.reshape(2, 3);
        System.out.println(reshape);
        // 按行进行归一化，得到的是系数
        System.out.println(reshape.norm2(true,1));
        System.out.println(reshape);
        // ND4J中的广播API
        System.out.println(reshape.divColumnVector(reshape.norm2(true, 1)));
    }
    //Implement a softmax function using numpy. You can think of softmax as a normalizing function used when your algorithm needs to classify two or more classes. You will learn more about softmax in the second course of this specialization.
    @Test
    public void test6(){
        INDArray rand = Nd4j.createFromArray(9.0,2.0,5.0,0.0,0.0,7.0,5.0,0.0,0.0,0.0 );
        INDArray reshape = rand.reshape(2, 5);
        System.out.println(reshape);
        // 按行进行归一化，得到的是系数
        System.out.println(softMax(reshape));
    }

//Implement the numpy vectorized version of the L1 loss. You may find the function abs(x) (absolute value of x) useful.
    @Test
    public void test7(){
        INDArray yHat = Nd4j.createFromArray(.9, 0.2, 0.1, .4, .9);
        INDArray y = Nd4j.createFromArray(1.0, 0.0 , 0.0 , 1.0 , 1.0 );

        System.out.println(loss(yHat,y));
    }
//Implement the numpy vectorized version of the L2 loss
    @Test
    public void test8(){
        INDArray yHat = Nd4j.createFromArray(.9, 0.2, 0.1, .4, .9);
        INDArray y = Nd4j.createFromArray(1.0, 0.0 , 0.0 , 1.0 , 1.0 );

        System.out.println(loss2(yHat,y));
    }
    public static double sigmoidOne(double x){

        return 1.0/(Math.exp(-x)+1);
    }

    public static INDArray sigmoidTwo(INDArray x){
        
        return Nd4j.math.pow(Nd4j.math().exp(x.mul(-1)).add(1),-1.0);
    }

    public static INDArray sigmoidGrad(INDArray x){
        INDArray s = Nd4j.math.pow(Nd4j.math().exp(x.mul(-1)).add(1), -1.0);

        INDArray ds = s.mul(s.mul(-1).add(1));
        return ds;

    }

    public static INDArray image2vector(INDArray image){
        return image.reshape(image.shape()[0] * image.shape()[1] * image.shape()[2],1);
    }


    public static INDArray softMax(INDArray x){
        INDArray exp = Nd4j.math().exp(x);

        INDArray sum = exp.sum(true, 1);

        INDArray broadcast = sum.broadcast(x.shape()[0], x.shape()[1]);
        INDArray resu = x.div(broadcast);
        return resu;
    }
    public static INDArray loss(INDArray yHat,INDArray y){
        
        INDArray newY = yHat.sub(y);
        INDArray absolutNewY = Nd4j.math().abs(newY);
        INDArray sum = absolutNewY.sum(true, 0);


        return sum;

    }

    public static INDArray loss2(INDArray yHat,INDArray y){
        INDArray newY = yHat.sub(y);
        INDArray absolutNewY = Nd4j.math().pow(newY,2.0);
        INDArray sum = absolutNewY.sum(true, 0);
        return sum;
    }

}
