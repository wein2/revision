package avp;


import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class avpiterator {

    private static int height = 256;
    private static int width = 256;
    private static int nChannels = 3;
    private static int nLabels = 2;

    private static String[] allowedExt = BaseImageLoader.ALLOWED_FORMATS;
    private static int trainRatio = 70;
    private static int batchSize;
    private static int seed = 123;
    private static Random rng = new Random();
    private static ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
    private static InputSplit trainData, testData;
    private static ImageTransform transform = null;
    private static DataNormalization scaler = new ImagePreProcessingScaler(0,1);

    public avpiterator() throws IOException
    {

    }

    static void setup() throws IOException
    {

        File input = new ClassPathResource("data\\train").getFile();
        FileSplit filesplit = new FileSplit(input,allowedExt,rng);
        BalancedPathFilter bpf = new BalancedPathFilter(rng,allowedExt,labelMaker);

        // split the image into train and test
        InputSplit[] allData = filesplit.sample(bpf,trainRatio,100-trainRatio);
        testData = allData[1];
        trainData = allData[0];


    }

    private static DataSetIterator makeIterator(boolean train) throws IOException {

        ImageRecordReader rr = new ImageRecordReader(height,width, nChannels,labelMaker);
        if(train && transform != null)
            rr.initialize(trainData,transform);
        else if(train)
            rr.initialize(trainData);
        else
            rr.initialize(testData);

        DataSetIterator iter = new RecordReaderDataSetIterator(rr,batchSize,1,2);
        iter.setPreProcessor(scaler);

        return iter;
    }

    public static DataSetIterator getTrain(int batchsize, ImageTransform imgtrans) throws IOException {
        batchSize = batchsize;
        transform = imgtrans;
        return makeIterator(true);


    }
    public static DataSetIterator getTest(int batchsize) throws IOException {
        batchSize = batchsize;
        return makeIterator(false);
    }




}

