package question2;

import org.datavec.image.transform.*;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.core.storage.StatsStorageListener;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;

import java.io.IOException;
import java.util.Random;

/* ===================================================================
 * We will solve a task of classifying horse breeds.
 * The dataset contains 4 classes, each with just over 100 images
 * Images are of 256x256 RGB
 *
 * Source: https://www.kaggle.com/olgabelitskaya/horse-breeds
 * ===================================================================
 * TO-DO
 *
 * 1. In HorseBreedIterator complete both methods (i) setup and (ii) makeIterator
 * 2. Complete ImageTransform Pipeline
 * 3. Complete your network configuration
 * 4. Train your model and set listeners
 * 5. Perform evaluation on both train and test set
 * 6. [OPTIONAL] Mitigate the underfitting problem
 *
 * ====================================================================
 * Assessment will be based on
 *
 * 1. Correct and complete configuration details
 * 2. HorseBreedClassifier is executable
 * 3. Convergence of the network
 *
 * ====================================================================
 ** NOTE: Only make changes at sections with the following. Replace accordingly.
 *
 *   /*
 *    *
 *    * WRITE YOUR CODES HERE
 *    *
 *    *
 */

public class HorseBreedClassifier {

    private static final Logger log = org.slf4j.LoggerFactory.getLogger(HorseBreedClassifier.class);
    private static final int height = 64;
    private static final int width = 64;
    private static final int nChannel = 3;
    private static final int nOutput = 4;
    private static final int seed = 141;
    private static Random rng = new Random(seed);
    private static double lr = 1e-4;
    private static final int nEpoch = 20;
    private static final int batchSize = 3;

    public static void main(String[] args) throws IOException {

        HorseBreedIterator.setup();

        // Build an Image Transform pipeline consisting of
        // a horizontal flip, crop, rotation, and random cropping



        ImageTransform transform = new PipelineImageTransform.Builder()
                .addImageTransform(new FlipImageTransform(1),0.3)
                .addImageTransform(new CropImageTransform(25))
                .addImageTransform(new RotateImageTransform(25))
                .addImageTransform(new CropImageTransform(rng,25))
                .build();


        DataSetIterator trainIter = HorseBreedIterator.getTrain(transform,batchSize);
        DataSetIterator testIter = HorseBreedIterator.getTest(1);

        // Build your model configuration

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.RELU)
                .updater(new Adam(lr))
                .convolutionMode(ConvolutionMode.Same)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .l2(5*1e-4)
                .list()
                .layer(0,new ConvolutionLayer.Builder(new int[] {11,11},new int[]{4,4})
                    .name("cnn1")
                    .convolutionMode(ConvolutionMode.Truncate)
                    .nIn(nChannel)
                    .nOut(96)
                    .build())
                .layer(1,new LocalResponseNormalization.Builder().build())
                .layer(2,new SubsamplingLayer.Builder(
                        SubsamplingLayer.PoolingType.AVG)
                    .kernelSize(2,2)
                    .stride(1,1)
                    .padding(1,1)
                    .name("avgpool1")
                    .build())
                .layer(3,new ConvolutionLayer.Builder(new int[]{5,5}, new int[]{1,1},new int[]{2,2})
                    .name("cnn2")
                    .convolutionMode(ConvolutionMode.Truncate)
                    .nOut(256)
                    .biasInit(0.1)
                    .build())
                .layer(4,new SubsamplingLayer.Builder(
                        SubsamplingLayer.PoolingType.AVG, new int[]{5,5},
                        new int[]{2,2})
                    .name("avgpool2")
                    .convolutionMode(ConvolutionMode.Truncate)
                    .build())
                .layer(5, new DenseLayer.Builder()
                    .name("ffn1")
                    .nOut(1024)
                    .weightInit(WeightInit.XAVIER)
                    .biasInit(0.1)
                    .dropOut(0.2)
                    .build())
                .layer(6,new DenseLayer.Builder()
                    .name("ffn2")
                    .nOut(1024)
                    .weightInit(WeightInit.XAVIER)
                    .biasInit(0.1)
                    .dropOut(0.1)
                    .build())
                .layer(7, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                    .name("output")
                    .nOut(nOutput)
                    .activation(Activation.SOFTMAX)
                    .weightInit(WeightInit.XAVIER)
                    .build())
                    .setInputType(InputType.convolutional(height,width,nChannel))
                .setInputType(InputType.convolutional(height,width,nChannel))
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();

        log.info("**************************************** MODEL SUMMARY ****************************************");
        System.out.println(model.summary());

        // Train your model and set listeners
        UIServer uiserver = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiserver.attach(statsStorage);

        model.setListeners(new StatsListener(statsStorage,10));
        model.fit(trainIter,nEpoch);
        /*
         *
         *
         *  Write your codes here
         *
         *
         */
        log.info(model.summary());
        log.info("**************************************** MODEL EVALUATION ****************************************");

        // Perform evaluation on both train and test set

        /*
         *
         *
         *  Write your codes here
         *
         *
         */
        Evaluation evalTrain,evalTest;
        evalTrain = model.evaluate(trainIter);
        evalTest = model.evaluate(testIter);
        System.out.println("=============== Evaluation on training set =======================\n");
        System.out.println(evalTrain.stats());
        System.out.println("=============== Evaluation on testing set =======================\n");
        System.out.println(evalTest.stats());
    }

}
