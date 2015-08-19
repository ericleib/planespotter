package tk.thebrightstuff;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import org.openimaj.data.DataSource;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.io.IOUtils;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator.Mode;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.util.pair.IntFloatPair;

import de.bwaldvogel.liblinear.SolverType;



public class AircraftApp {

	public LiblinearAnnotator<FImage, String> ann;
	
	public void build (List<Aircraft> trainingDataset, int nbQuantiser, int nclass) throws Exception {
		
		System.out.println("Creating dense SIFT feature extractor...");
		DenseSIFT dsift = new DenseSIFT(5, 7);
		PyramidDenseSIFT<FImage> pdsift = new PyramidDenseSIFT<FImage>(dsift, 6f, 7);
		
		System.out.println("Training quantiser (using training dataset)...");
		HardAssigner<byte[], float[], IntFloatPair> assigner = 
				trainQuantiser(AircraftDataset.sample(trainingDataset, nbQuantiser, nclass), pdsift);

		System.out.println("New PHOW assigner...");
		//FeatureExtractor<DoubleFV, Record<FImage>> extractor = new PHOWExtractor(pdsift, assigner);
		FeatureExtractor<DoubleFV, FImage> extractor = new PHOWExtractorFImage(pdsift, assigner);
				
		System.out.println("Training classifier...");
		//LiblinearAnnotator<Record<FImage>, String> ann = new LiblinearAnnotator<Record<FImage>, String>(
	    //        extractor, Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
		//ann = new LiblinearAnnotator<FImage, String>(
	    //        extractor, Mode.MULTILABEL, SolverType.L2R_LR_DUAL, 1.0, 0.00001);
		ann = new LiblinearAnnotator<FImage, String>(
			      extractor, Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
		
		ann.train(trainingDataset);
		
	}
	
	public CMResult<String> test(Map<FImage,Set<String>> dataset){

		System.out.println("New classifier tester (Using test dataset)...");
		ClassificationEvaluator<CMResult<String>, String, FImage> eval = 
				new ClassificationEvaluator<CMResult<String>, String, FImage>(
					ann, dataset, new CMAnalyser<FImage, String>(CMAnalyser.Strategy.SINGLE));

		System.out.println("Running tester...");
		Map<FImage, ClassificationResult<String>> guesses = eval.evaluate();

		System.out.println("Analysing results...");
		CMResult<String> result = eval.analyse(guesses);

		System.out.println("Detailed report:");
		System.out.println(result.getDetailReport());
		
		return result;
	}
	
	public void classify(FImage image){

		System.out.println();
		System.out.println("Classifying a picture...");
		ClassificationResult<String> res = ann.classify(image);

		System.out.println("Labels:");
		for(String str : res.getPredictedClasses())
			System.out.println(" - "+str+" ("+res.getConfidence(str)+")");
		
	}
	
	public void saveToFile(File f) throws IOException {
		IOUtils.writeToFile(ann, f);
	}
	
	public void restoreFromFile(File f) throws IOException {
		ann = IOUtils.readFromFile(f);
	}
	
	public static void main(String[] args) throws Exception {

		Path p = Paths.get("C:\\Users\\niluje\\Documents\\planespotter");
		Path f = Paths.get("planes-airliners-filtered2.txt");
		File datafile = p.resolve("data.txt").toFile();

		int factor = 700, nclass = 7;
		
		System.out.println("Getting data...");
		AircraftDataset dataset = new AircraftDataset( p, f, Aircraft.Annotator.MANUF_MODEL );
		dataset.readAircraft();
		System.out.println("Size before filtering: "+ dataset.size());
		dataset.filter(AircraftDataset.MANUF, new HashSet<String>(Arrays.asList(new String[]{"boeing","airbus"})));
		dataset.filter(AircraftDataset.MODEL, new HashSet<String>(Arrays.asList(new String[]{"a320","a380","a330-300","777-200","747-400","737-800","787-8"})));
		System.out.println("Size after filtering: "+ dataset.size());
		
		List<Aircraft> sample = AircraftDataset.sample(dataset.aircraft, 100 * factor, nclass);
		System.out.println("Size sample: "+ sample.size());
		List<Aircraft> training = AircraftDataset.sample(sample, 90 * factor, nclass);
		System.out.println("Size training: "+ training.size());
		List<Aircraft> validating = AircraftDataset.sample(training, 10 * factor, nclass);
		System.out.println("Size validating: "+ validating.size());
		List<Aircraft> testing = AircraftDataset.negative(sample, training);
		System.out.println("Size testing: "+ testing.size());
		training = AircraftDataset.negative(training, validating);
		System.out.println("Size training: "+ training.size());

		System.out.println("Training dataset: ");
		AircraftDataset.stats(training).forEach(System.out::println);
		System.out.println("Test dataset: ");
		AircraftDataset.stats(testing).forEach(System.out::println);


		System.out.println("Writing datasets:");
		Files.write(p.resolve("training.txt"), training.stream().map(ac->ac.toString()).collect(Collectors.toList()));
		Files.write(p.resolve("testing.txt"), testing.stream().map(ac->ac.toString()).collect(Collectors.toList()));
		Files.write(p.resolve("validating.txt"), validating.stream().map(ac->ac.toString()).collect(Collectors.toList()));
		
		AircraftApp app = new AircraftApp();
		
		
		if( datafile.exists() ){

			System.out.println("Restoring from file...");
			app.restoreFromFile(datafile);
			
		}else{
			
			System.out.println("Building app...");
			app.build(training, 100 * factor, nclass);

			System.out.println("Saving to file...");
			app.saveToFile(datafile);

		}

		System.out.println("Testing app...");
		app.test(AircraftDataset.toMap(testing));
		
	}
	
	

	private static HardAssigner<byte[], float[], IntFloatPair> 
			trainQuantiser(List<Aircraft> sample, PyramidDenseSIFT<FImage> pdsift)
	{
		List<LocalFeatureList<ByteDSIFTKeypoint>> allkeys = new ArrayList<LocalFeatureList<ByteDSIFTKeypoint>>();

		System.out.println(" Sample size: "+sample.size());
		//List<FImage> images = new ArrayList<FImage>();
		int i=0;
		for (Aircraft ac : sample) {
			System.out.print("  Image #"+(i++));
			FImage img = ac.getObject();
			//images.add(img);
			pdsift.analyseImage(img);
			LocalFeatureList<ByteDSIFTKeypoint> keypoints = pdsift.getByteKeypoints(0.005f);
			allkeys.add(keypoints);
			System.out.println("  Key points number = "+keypoints.size());
		}
		
		//DisplayUtilities.display("Images used to train quantiser: ", images);

		if (allkeys.size() > 10000)
			allkeys = allkeys.subList(0, 10000);

		System.out.println(" createKDTreeEnsemble");
		ByteKMeans km = ByteKMeans.createKDTreeEnsemble(300);
		System.out.println(" LocalFeatureListDataSource");
		DataSource<byte[]> datasource = new LocalFeatureListDataSource<ByteDSIFTKeypoint, byte[]>(allkeys);
		System.out.println(" cluster");
		ByteCentroidsResult result = km.cluster(datasource);

		System.out.println(" defaultHardAssigner");
		return result.defaultHardAssigner();
	}
	
	
	
	
	private static class PHOWExtractorFImage implements FeatureExtractor<DoubleFV, FImage> {
	    PyramidDenseSIFT<FImage> pdsift;
	    HardAssigner<byte[], float[], IntFloatPair> assigner;

	    public PHOWExtractorFImage(PyramidDenseSIFT<FImage> pdsift, HardAssigner<byte[], float[], IntFloatPair> assigner)
	    {
	        this.pdsift = pdsift;
	        this.assigner = assigner;
	    }

	    public DoubleFV extractFeature(FImage image) {

	        pdsift.analyseImage(image);

	        BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<byte[]>(assigner);

	        BlockSpatialAggregator<byte[], SparseIntFV> spatial = new BlockSpatialAggregator<byte[], SparseIntFV>(
	                bovw, 2, 2);

	        return spatial.aggregate(pdsift.getByteKeypoints(0.015f), image.getBounds()).normaliseFV();
	    }
	}
	
	
}
