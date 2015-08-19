package tk.thebrightstuff;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.openimaj.image.FImage;


public class AircraftDataset {
	
	private static final Pattern LINE = Pattern.compile("\"(.*)\" \"(.*)\" \"(.*)\" \"(.*)\"");
	
	public static final int MANUF = 0, MODEL = 1, AIRLINE = 2;

	public final Map<String,List<Aircraft>> models = new HashMap<>();
	public final Map<String,List<Aircraft>> manufs = new HashMap<>();
	public final Map<String,List<Aircraft>> airlines = new HashMap<>();
	public final List<Aircraft> aircraft = new ArrayList<>();

	private final Path path, db;
	
	private final Aircraft.Annotator annotator;
	
	
	public AircraftDataset(Path path, Path db, Aircraft.Annotator annotator){
		this.path = path;
		this.db = db;
		this.annotator = annotator;
	}
	
	private AircraftDataset(List<Aircraft> aircraft){
		this.aircraft.addAll(aircraft);
		path = db = null;
		annotator = null;
	}
	
	public void readAircraft() throws IOException {

		File pathf = path.toFile();
		Files.readAllLines(path.resolve(db)).forEach((str) -> {
			Matcher m = LINE.matcher(str);
			if(m.matches()){
				aircraft.add(new Aircraft(pathf, m.group(1), m.group(2), m.group(3), m.group(4), annotator));
			}else{
				System.out.println("Line: "+str+" does not match!");
			}
		});
		
	}
	

	public void updateMaps() {
		manufs.clear();
		models.clear();
		airlines.clear();
		aircraft.forEach((a) -> {
			add(manufs, a.manuf, a);
			add(models, a.model, a);
			add(airlines, a.airline, a);
		});
	}
	
	private void add(Map<String, List<Aircraft>> map, String key, Aircraft a) {
		if(!map.containsKey(key))
			map.put(key, new ArrayList<Aircraft>());
		map.get(key).add(a);
	}
	


	public void filter(int nb, Set<String> list) {

		switch(nb){
		case MANUF: aircraft.removeIf((ac) -> ! list.contains(ac.manuf)); return;
		case MODEL: aircraft.removeIf((ac) -> ! list.contains(ac.model)); return;
		case AIRLINE: aircraft.removeIf((ac) -> ! list.contains(ac.airline)); return;
		}
		
	}
	
	
	public List<String> stats() {
		
		updateMaps();
		
		List<String> lines = new ArrayList<>();
		lines.add("*** STATISTICS ***");
		lines.add("MANUFACTURERS");
		lines.addAll(mapStats(manufs.entrySet().stream().sorted(Aircraft.comparator)));
		
		lines.add("");
		lines.add("AIRCRAFT");
		lines.addAll(mapStats(models.entrySet().stream().sorted(Aircraft.comparator)));

		lines.add("");
		lines.add("AIRLINES");
		lines.addAll(mapStats(airlines.entrySet().stream().sorted(Aircraft.comparator)));
		
		return lines;
		
	}
	
	public int size(){
		return aircraft.size();
	}

	private List<String> mapStats(Stream<Entry<String, List<Aircraft>>> stream) {
		return stream
				.map((e) -> e.getKey()+whiteSpaces(e.getKey(),Integer.toString(e.getValue().size()))+e.getValue().size())
				.collect(Collectors.toList());
	}

	public void writeToFile(Path p) throws IOException {
		Files.write(path, aircraft.stream().map((ac) -> ac.toString()).collect(Collectors.toList()));
	}

	public Path getPath(){
		return path;
	}
	
	private static String whiteSpaces(String str, String str2) {
		int total = 50;
		StringBuilder b = new StringBuilder();
		for(int i=0; i<total-str.length()-str2.length(); i++)
			b.append(" ");
		return b.toString();
	}


	public static List<Aircraft> sample(List<Aircraft> aircraft, int size, int nclass) throws Exception {
		int i = new Random().nextInt(aircraft.size());
		List<Aircraft> list = new ArrayList<Aircraft>();
		Map<String,Integer> map = new HashMap<String,Integer>();
		while(list.size() < size){
			if(i == aircraft.size()) i = 0;
			Aircraft ac = aircraft.get(i++);
			String label = ac.getAnnotations().get(0);
			if(nclass == 0 || ((!map.containsKey(label)) && map.size()<nclass) || (map.containsKey(label) && map.get(label) < ((float) size) / ((float) nclass))){
				if(list.contains(ac))
					throw new Exception("Duplicate !");
				map.put(label, map.getOrDefault(label, 0)+1);
				list.add(ac);
			}				
		}
		return list;
	}


	public static List<Aircraft> negative(List<Aircraft> listA, List<Aircraft> listB) {
		return listA.stream()
				.filter((ac) -> !listB.contains(ac))
				.collect(Collectors.toList());
	}


	public static Map<FImage, Set<String>> toMap(List<Aircraft> list) {
		Map<FImage,Set<String>> map = new HashMap<FImage,Set<String>>();
		list.forEach((ac) -> map.put(ac.getObject(), ac.getAnnotations().stream().collect(Collectors.toSet())));
		return map;
	}


	public static List<String> stats(List<Aircraft> list) {
		return new AircraftDataset(list).stats();
	}

}
