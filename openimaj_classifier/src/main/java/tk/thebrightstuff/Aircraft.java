package tk.thebrightstuff;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Map.Entry;

import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.ml.annotation.Annotated;

public class Aircraft implements Annotated<FImage,String> {
	
	public static final Comparator<Entry<String, List<Aircraft>>> comparator = 
			(o1, o2) -> o2.getValue().size()-o1.getValue().size();
			
	public enum Annotator {
		
		MULTILABEL {
			@Override
			public String[] annot(Aircraft ac) {
				return new String[]{ac.manuf, ac.model, ac.airline};
			}
		},
		MANUF {
			@Override
			public String[] annot(Aircraft ac) {
				return new String[] {ac.manuf};
			}
		},
		MANUF_MODEL {
			@Override
			public String[] annot(Aircraft ac) {
				return new String[] {ac.manuf+" "+ac.model};
			}
		};
				
		public abstract String[] annot(Aircraft ac);
	}
		
	public final File db;
	public final String path, manuf, model, airline;
	public final Annotator annotator;
	
	public Aircraft(File db, String path, String manuf, String model, String airline, Annotator annotator){
		this.db = db;
		this.path = path;
		this.manuf = manuf;
		this.model = model;
		this.airline = airline;
		this.annotator = annotator;
	}
	
	
	public String toString(){
		return "\""+path+"\" \""+manuf+"\" \""+model+"\" \""+airline+"\"";
	}


	@Override
	public FImage getObject() {
		try {
			return ImageUtilities.readF(new File(db, path));
		} catch (IOException e) {
			e.printStackTrace();
			return null;
		}
	}


	@Override
	public List<String> getAnnotations() {
		return Arrays.asList(annotator.annot(this));
	}

}
