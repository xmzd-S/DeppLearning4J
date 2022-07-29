package com.xiumuzidiao.dp.course1Week2;

import java.net.IDN;
import java.util.Arrays;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class TestDemo2{
    public static void main(String[] args) {
        INDArray i = Nd4j.createFromArray(0.0,2.0);
        System.out.println(Demo2.sigmoid(i));

        System.out.println(Arrays.toString(Demo2.initWAndB(2)));
        System.out.println("=======================================================");
        // w矩阵1*2  x矩阵 2*3   b 矩阵 1*3   y矩阵 1*3   
        INDArray wt = Nd4j.createFromArray(1.,2.).reshape(2, 1).transpose();
        
        INDArray b = Nd4j.createFromArray(2.,2.,2.).reshape(1, 3);
        INDArray x = Nd4j.createFromArray(new Double[][]{   {1.,2.,-1.},{3.,4.,-3.2}});
       
        INDArray y = Nd4j.createFromArray(1.,0.,1.).reshape(1, 3);
        System.out.println(Arrays.toString(Demo2.propagate(wt, b, x, y)));

        System.out.println(Arrays.toString(Demo2.propagateTwo(wt, b, x, y)));

        Object[] o = Demo2.propagateThree(wt, b, x, y);
        INDArray I = (INDArray)o[1];
        System.out.println(Arrays.toString(I.shape()));



        // 迭代测试
        INDArray wt1 = Nd4j.createFromArray(0.1124579,0.23106775).reshape(2, 1).transpose();
        
        INDArray b1 = Nd4j.createFromArray(-0.3,-0.3,-0.3).reshape(1, 3);
        INDArray x1 = Nd4j.createFromArray(new Double[][]{   {1. , -1.1, -3.2},{1.2,  2. ,  0.1}});
       
        INDArray y1 = Nd4j.createFromArray(1.,0.,1.).reshape(1, 3);
        Demo2.optimize(wt1, b1, x1, y1, 1000, 0.001, true);


    }
}