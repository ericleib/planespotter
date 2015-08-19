package tk.thebrightstuff;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashSet;
import java.util.Set;

import org.json.*;

public class JsonProcessor {

	private File file;

	public JsonProcessor(File file){
		this.file = file;
	}
	
	public void process(BufferedWriter dest, Set<String> allfiles) throws IOException 
	{
		System.out.println("Reading file "+file.getName());
		
		String str = readFile(file.getAbsolutePath(), StandardCharsets.UTF_8);
		System.out.println("File read");

		JSONArray ja = new JSONArray(str);
		System.out.println(ja.length()+" elements");
		
		for(int i=0; i<ja.length(); i++){
			JSONObject obj = ja.getJSONObject(i);
			if(obj.has("images") && obj.has("manuf") && obj.has("model") && obj.has("airline")){
				JSONArray imageArray = obj.getJSONArray("images");
				if(imageArray.length()!=1){
					System.err.println("Weird image object: \n"+imageArray.toString());
				}else{
					String path = imageArray
							.getJSONObject(0)
							.getString("path")
							.trim()
							.replaceAll("full/(\\w)(\\w)", "images/$1/$2/");
					String model = decompose(obj.getString("model").trim());
					String manuf = decompose(obj.getString("manuf").trim());
					String airline = decompose(obj.getString("airline").trim());
					
					if(!allfiles.contains(path)){
						if(manuf.equals("") || model.equals("") || airline.equals(""))
							System.out.println("Missing data: "+obj.toString());
						else{
							dest.write("\""+path+"\" \""+manuf+"\" \""+model+"\" \""+airline+"\"\n");
							allfiles.add(path);
						}
					}else
						System.out.println("file "+path+" already in DB");
				}
			}
		}
		System.out.println("Entries written to dest");
		System.out.println();
	}
	
	public static String readFile(String path, Charset encoding) throws IOException 
	{
		byte[] encoded = Files.readAllBytes(Paths.get(path));
		return new String(encoded, encoding);
	}
	
	public static String decompose(String s) {
	    return java.text.Normalizer.normalize(s, java.text.Normalizer.Form.NFD).replaceAll("\\p{InCombiningDiacriticalMarks}+","").replaceAll("[^\\x20-\\x7e]", "");
	}

	public static void main(String[] args) throws Exception {

		//System.out.println("full/066e07b60a5be4e627acc0b8da8441ffbf7cac09.jpg".replaceAll("full/(\\d)(\\d)", "images/$1/$2/"));
		
		Set<String> allfiles = new HashSet<String>();
		
		try(BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("C:\\Users\\niluje\\Documents\\planespotter\\planes-airliners.txt"),"UTF-8"))){

			new JsonProcessor(new File("C:\\Users\\niluje\\Documents\\planespotter\\planes-airliners-p1-p346.json")).process(writer, allfiles);
			new JsonProcessor(new File("C:\\Users\\niluje\\Documents\\planespotter\\planes-airliners-p346-p1236.json")).process(writer, allfiles);
			new JsonProcessor(new File("C:\\Users\\niluje\\Documents\\planespotter\\planes-airliners.json")).process(writer, allfiles);
			
		}
		
	}
}
