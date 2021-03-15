package datavec;

import org.apache.commons.io.FileUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.writer.RecordWriter;
import org.datavec.api.records.writer.impl.csv.CSVRecordWriter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.partition.NumberOfRecordsPartitioner;
import org.datavec.api.split.partition.Partitioner;
import org.datavec.api.transform.MathOp;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.condition.ConditionOp;
import org.datavec.api.transform.condition.column.DoubleColumnCondition;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.time.DeriveColumnsFromTimeTransform;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.joda.time.DateTimeFieldType;
import org.joda.time.DateTimeZone;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.ListIterator;

public class loadCSV {

    public static void main(String[] args) throws Exception
    {

      // ========================== define schema =====================
        Schema inputSchema = new Schema.Builder()
                .addColumnString("DateTimeString")
                .addColumnsString("CustomerID","MerchantID")
                .addColumnInteger("NumItemsInTransaction")
                .addColumnCategorical("MerchantCountryCode",
                        Arrays.asList("USA","CAN","FR","MX"))
                //addColumnDouble(String name, Double minAllowedVlue, Double maxAllowedValue, boolean allowedNaN,boolean allowedInfinite)
                .addColumnDouble("TransactionAmountUSD",null,10.0,false,false)
                .addColumnCategorical("FraudLabel",Arrays.asList("Fraud","Legit"))
                .build();

        System.out.println("Input data schema details: ");
        System.out.println(inputSchema);

        System.out.println("\n\nOther information obtainable :");
        System.out.println(inputSchema.getColumnNames());
        System.out.println(inputSchema.getColumnTypes());
        System.out.println(inputSchema.numColumns());

        // ====================== transformation =====================
        TransformProcess tp = new TransformProcess.Builder(inputSchema)
                .removeColumns("MerchantID","CustomerID")
                .conditionalReplaceValueTransform(
                        "TransactionAmountUSD",
                        new DoubleWritable(0.0),
                        new DoubleColumnCondition(
                                "TransactionAmountUSD",
                                ConditionOp.LessThan,0.0
                        )
                )
                .stringToTimeTransform("DateTimeString",
                        "YYYY-MM-DD HH:mm:ss.SSS", DateTimeZone.UTC)
                .renameColumn("DateTimeString","DateTime")
                .transform(new DeriveColumnsFromTimeTransform.Builder("DateTime")
                    .addIntegerDerivedColumn("HourOfDay",
                        DateTimeFieldType.hourOfDay())
                    .build()
                )
                .doubleMathOp("TransactionAmountUSD", MathOp.Divide, 0.0)
                .removeColumns("DateTime")
                .categoricalToInteger("MerchantCountryCode","FraudLabel")
                .build();

        Schema outputSchema = tp.getFinalSchema();

        System.out.println("\n\nSchema after transforming data: \n");
        System.out.println(outputSchema);

        // ======================= load data ==============================
        File inputFile = new ClassPathResource("datavec\\exampledata.csv").getFile();

        // define input reader
        RecordReader rr = new CSVRecordReader(0,',');
        rr.initialize(new FileSplit(inputFile));

        // PROCESS THE DATA
        List<List<Writable>> originalData = new ArrayList<>();
        while(rr.hasNext())
            originalData.add(rr.next());

        List<List<Writable>> processedData = LocalTransformExecutor.execute(originalData,tp);
        List<List<Writable>> processedData2 = new ArrayList<>();
        ListIterator iter = originalData.listIterator();
        while(iter.hasNext())
            processedData2.add(tp.execute((List<Writable>) iter.next()));


        System.out.println("\n\nOriginal Data: \n");
        System.out.println(originalData);
        System.out.println("\n\nTransformed Data: \n");
        System.out.println(processedData2);
        System.out.println(".................");

        // Create iterator from processedData
        RecordReader collectionRecordReader = new CollectionRecordReader(processedData);

        // RecordReaderDataSetIterator(RecordReader, int bathSize, int labelIndex, int numPossibleLabel)
        DataSetIterator iterator = new RecordReaderDataSetIterator(collectionRecordReader,1,4,2);

        // ====================== Write into file =========================
        File outputFile = new File("ExampleDataOutput.csv");
        if(outputFile.exists())
            outputFile.delete();
        outputFile.createNewFile();

        RecordWriter rw = new CSVRecordWriter();
        Partitioner p = new NumberOfRecordsPartitioner();
        rw.initialize(new FileSplit(outputFile),p);
        rw.writeBatch(processedData);
        rw.close();

        //Print before + after:
        System.out.println("\n\n---- Original Data File ----");

        String originalFileContents = FileUtils.readFileToString(inputFile, Charset.defaultCharset());
        System.out.println(originalFileContents);

        System.out.println("\n\n---- Processed Data File ----");
        String fileContents = FileUtils.readFileToString(outputFile, Charset.defaultCharset());
        System.out.println(fileContents);

        System.out.println("\n\nDONE");









    }



}
