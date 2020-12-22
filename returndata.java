import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Scanner;  

public class returndata {

	private static final String FILENAME = "data.txt";

	public static void main(String[] args) {
		String data="";
		BufferedReader br = null;
		FileReader fr = null;

			String sCurrentLine="";
		try {

			//br = new BufferedReader(new FileReader(FILENAME));
			fr = new FileReader(FILENAME);
			br = new BufferedReader(fr);


			while ((sCurrentLine = br.readLine()) != null) {
				//System.out.println(sCurrentLine);
				data=data + sCurrentLine;
			}

		} catch (IOException e) {

			e.printStackTrace();

		} finally {

			try {

				if (br != null)
					br.close();

				if (fr != null)
					fr.close();

			} catch (IOException ex) {

				ex.printStackTrace();

			}

		}
//System.out.println( sCurrentLine);


System.out.println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Sandeep Padhi");


String array2[]= data.split(",",-5);
       for (String temp: array2){
          System.out.println(temp);
       }



String value[]=data.split(",",0);

System.out.println("Length of sCurrentLine is "+ value.length);


Scanner sc=new Scanner(System.in);
//BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));

while(true){

String input=sc.next();
int index=0;

for(String i:value)
{
if (i.equals(input))
{
System.out.println("Element found at index "+ index);
break;
}

}


}

	}

}

