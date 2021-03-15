package avp;


import org.datavec.image.transform.*;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.EpochTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.termination.ScoreImprovementEpochTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.common.primitives.Pair;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

import java.util.*;
import java.util.concurrent.TimeUnit;

public class avpclassifier {

    protected static final Logger log = LoggerFactory.getLogger(avpclassifier.class);

    private static int seed = 123;
    private static Random rng = new Random(seed);
    private static int nEpoch = 20;
    private static int batchSize = 1;
    private static int nChannels = 3;
    private static DataSetIterator trainIter, testIter, transformedIter;

    public static void main(String[] args) throws IOException
    {
        log.info("Load data...");
        avpiterator.setup();

        testIter = avpiterator.getTest(1);

        // ========================== image transform =============================

        ImageTransform flipimage = new FlipImageTransform(-1);
        ImageTransform rotateimage20 = new RotateImageTransform(20);
        ImageTransform rotateimage40 = new RotateImageTransform(40);
        ImageTransform warpimage = new WarpImageTransform(rng,40);

        List<Pair<ImageTransform,Double>> pipeline = Arrays.asList(

                new Pair<>(flipimage,0.5),
                new Pair<>(rotateimage20,0.5),
                new Pair<>(rotateimage40,0.5),
                new Pair<>(warpimage,0.5)
        );

        PipelineImageTransform transform = new PipelineImageTransform(pipeline,true);

/*
        ImageTransform transform = new PipelineImageTransform.Builder()
                .addImageTransform(new FlipImageTransform(-1),0.5)
                .addImageTransform(new RotateImageTransform(20),0.5)
                .addImageTransform(new RotateImageTransform(40),0.5)
                .addImageTransform(new WarpImageTransform(rng,40),0.5)
                .build();
*/

        // trainIter = avpiterator.getTrain(batchSize,null);
        transformedIter = avpiterator.getTrain(batchSize,transform);

        // data augmentation
        // no need to do this step cause introducing image transform into
        // the iterator has already achieve the purpose of data augmentation
        /*        ArrayList<DataSet> allDataList = new ArrayList<DataSet>();
        while(trainIter.hasNext())
            allDataList.add(trainIter.next());
        System.out.println("Size of training set :");
        System.out.println(DataSet.merge(allDataList).numExamples());
        while(transformedIter.hasNext())
            allDataList.add(transformedIter.next());
        DataSet allData = DataSet.merge(allDataList);
        System.out.println("After data augmentation : ");
        System.out.println(allData.numExamples());
        allData.shuffle();
        DataSetIterator allDataIterator = new ViewIterator(allData,batchSize);*/
        

        // ============================= Network ================================

        MultiLayerConfiguration nnconfig = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.RELU)
                .updater(new Adam(1e-4))
                .convolutionMode(ConvolutionMode.Same)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .l2(5*1e-4)
                .list()
                .layer(0, new ConvolutionLayer.Builder(new int[] {3,3}, new int[] {1,1})
                    .name("cnn1")
                    .convolutionMode(ConvolutionMode.Truncate)
                    .nIn(nChannels)
                    .nOut(96)
                    .build())
                .layer(1,new LocalResponseNormalization.Builder()
                    .build())
                .layer(2, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG)
                    .kernelSize(2,2)
                    .stride(1,1)
                    .name("avgpool1")
                    .build())
                .layer(3,new ConvolutionLayer.Builder(new int[] {5,5}, new int[] {2,2})
                    .name("cnn2")
                    .convolutionMode(ConvolutionMode.Truncate)
                    .nOut(256)
                    .biasInit(0.1)
                    .build())
                .layer(4,new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                    .name("maxpool1")
                    .kernelSize(4,4)
                    .stride(1,1)
                    .build())
                .layer(5,new DenseLayer.Builder()
                    .name("ffn1")
                    .nOut(1024)
                    .weightInit(WeightInit.XAVIER)
                    .biasInit(0.1)
                    .dropOut(0.4)
                    .build())
                .layer(6,new DenseLayer.Builder()
                    .name("ffn2")
                    .nOut(512)
                    .weightInit(WeightInit.XAVIER)
                    .biasInit(0.1)
                    .dropOut(0.1)
                    .build())
                .layer(7, new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                    .name("output")
                    .nOut(2)
                    .activation(Activation.SIGMOID)
                    .weightInit(WeightInit.XAVIER)
                    .build())
                .setInputType(InputType.convolutional(256,256,3)
                )
                .build();

        // ======================= Early Stopping Config ==========================
        LocalFileModelSaver saver = new LocalFileModelSaver("model\\earlystopping");

        List<EpochTerminationCondition> eslist = Arrays.asList(
                new MaxEpochsTerminationCondition(50),
                new ScoreImprovementEpochTerminationCondition(3)
        );
        EarlyStoppingConfiguration esconfig = new EarlyStoppingConfiguration.Builder()
                .epochTerminationConditions(eslist)
                .evaluateEveryNEpochs(1)
                .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(500, TimeUnit.MINUTES))
                .scoreCalculator(new DataSetLossCalculator(testIter,true))
                .modelSaver(saver)
                .build();

        log.info("Earlystopping training to find best model...");
        EarlyStoppingTrainer estrainer = new EarlyStoppingTrainer(esconfig,nnconfig,transformedIter);
        EarlyStoppingResult<MultiLayerNetwork> result = estrainer.fit();

        // best model
        MultiLayerNetwork model = result.getBestModel();

        log.info("Print score v.s. epoch...");
        Map<Integer,Double> scorevsEpoch = result.getScoreVsEpoch();
        List<Integer> list = new ArrayList<>(scorevsEpoch.keySet());
        Collections.sort(list);
        for(Integer i : list)
            log.info(i + "\t" + scorevsEpoch.get(i));

        // =========================== Evaluation =======================
        Evaluation evalTrain,evalTest;
        evalTrain = model.evaluate(transformedIter);
        log.info("Statistics for training Data:");
        log.info(evalTrain.stats());
        evalTest = model.evaluate(testIter);
        log.info("Statistics for testing Data:");
        log.info(evalTest.stats());




    }

}
