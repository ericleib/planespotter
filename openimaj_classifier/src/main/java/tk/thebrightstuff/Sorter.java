package tk.thebrightstuff;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.apache.commons.compress.archivers.ArchiveException;
import org.apache.commons.compress.archivers.ArchiveStreamFactory;
import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;

public class Sorter {

	private String src, dest;
	
	private int i;

	public Sorter(String src, String dest){
		this.src = src;
		this.dest = dest;
	}
	
	public void sort() throws IOException, ArchiveException {
		
		i = 0;
		
		Path srcpath = Paths.get(src);
		if(Files.isDirectory(srcpath)) {

			try (DirectoryStream<Path> stream = Files.newDirectoryStream(srcpath)) {
	            for (Path path : stream) {
	            	if(Files.isRegularFile(path)){
	                	process(path.getFileName().toString(), Files.newInputStream(path));
	            	}
	            }
	        }
			
		} else {
			
		    InputStream is = new FileInputStream(srcpath.toString()); 
		    try(TarArchiveInputStream debInputStream = (TarArchiveInputStream) new ArchiveStreamFactory().createArchiveInputStream("tar", is)){
			    TarArchiveEntry entry = null; 
			    while ((entry = (TarArchiveEntry)debInputStream.getNextEntry()) != null) {
			        if (entry.isFile() && debInputStream.canReadEntryData(entry)) {
			        	process(new File(entry.getName()).getName(), debInputStream);
			        }
			    }
		    } 
			
		}		
		
		System.out.println(i+" files copied");
		
	}
	
	private void process(String name, InputStream in) throws IOException {
    	String str1 = name.substring(0, 1);
    	String str2 = name.substring(1, 2);
    	Path newDir = Paths.get(dest, str1, str2);
    	if(Files.notExists(newDir)){
    		System.out.println("Directory \""+newDir.toString()+"\" does not exist! (mkdir)");
    		Files.createDirectories(newDir);
    	}
    	//else{
    	//	System.out.println("Directory \""+newDir.toString()+"\" exists! "+str1+" "+str2);
    	//}
    	Path newPath = newDir.resolve(name.substring(2));
    	if(Files.notExists(newPath)){
        	Files.copy(in, newPath);
        	i++;
        	if( i%500 == 0 )
        		System.out.println(i+" files copied");
    	}else
    		System.out.println("File already exists! "+newPath.toString());
	}

	public static void main(String[] args) throws Exception {
		if(args.length!=2)
			throw new IOException("Expecting two directory paths as only arguments");
		
		new Sorter(args[0], args[1]).sort();
	}
}
