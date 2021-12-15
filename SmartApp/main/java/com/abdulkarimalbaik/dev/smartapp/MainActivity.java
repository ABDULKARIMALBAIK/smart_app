package com.abdulkarimalbaik.dev.smartapp;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Build;
import android.os.Environment;
import android.provider.MediaStore;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.support.v7.widget.Toolbar;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;

import static com.googlecode.javacv.cpp.opencv_highgui.*;
import static com.googlecode.javacv.cpp.opencv_core.*;
import static com.googlecode.javacv.cpp.opencv_imgproc.*;
import static com.googlecode.javacv.cpp.opencv_contrib.*;


import java.io.File;
import java.io.FileOutputStream;
import java.io.FilenameFilter;
import java.util.UUID;

public class MainActivity extends AppCompatActivity {

    private static final int PICK_CAPTURE_PHOTO = 1000;
    private static final int REQUEST_WRITE_READ_CAMERA_PICK = 1001;
    Button btnObjects , btnHumans , btnCanny , btnFace;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Toolbar toolbar = (Toolbar) findViewById(R.id.toolbar);
        toolbar.setTitle("Smart App");
        toolbar.setTitleTextColor(getResources().getColor(android.R.color.white));
        setSupportActionBar(toolbar);

        btnObjects = (Button)findViewById(R.id.btnObject);
        btnHumans = (Button)findViewById(R.id.btnHumans);
        btnCanny = (Button)findViewById(R.id.btnCanny);
        btnFace = (Button)findViewById(R.id.btnFace);

        btnObjects.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                startActivity(new Intent(MainActivity.this , ObjectActivity.class));
            }
        });


        btnHumans.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                startActivity(new Intent(MainActivity.this , HumanActivity.class));
            }
        });


        btnCanny.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                startActivity(new Intent(MainActivity.this , CannyActivity.class));
            }
        });

        btnFace.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

            }
        });

    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        switch (requestCode){
            case REQUEST_WRITE_READ_CAMERA_PICK:{

                if (grantResults[0] == PackageManager.PERMISSION_GRANTED &&
                        grantResults[1] == PackageManager.PERMISSION_GRANTED &&
                        grantResults[2] == PackageManager.PERMISSION_GRANTED){

                    Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(intent , PICK_CAPTURE_PHOTO);
                }
                else
                    Toast.makeText(this, "You can't detected your face !!!", Toast.LENGTH_SHORT).show();
                break;
            }
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == PICK_CAPTURE_PHOTO && resultCode == RESULT_OK && data != null && data.getExtras() != null){

            Bundle b = data.getExtras();
            Bitmap bitmap = (Bitmap)b.get("data");
            recognizerFace(bitmap);
        }
    }

    public void recognizerFace(Bitmap bitmap){

        File fileName = createFile(bitmap);


        String trainingDir = Environment.getExternalStorageDirectory() + "/dnns" ;
        CvArr testImage = cvLoadImage(fileName.getAbsolutePath());

        File root = new File(trainingDir);

        FilenameFilter jpgFilter = new FilenameFilter() {
            public boolean accept(File dir, String name) {
                return name.toLowerCase().endsWith(".jpg");
            }
        };

        File[] imageFiles = root.listFiles(jpgFilter);

        MatVector images = new MatVector(imageFiles.length);

        int[] labels = new int[imageFiles.length];

        int counter = 0;
        int label;

        CvArr img;  //
        CvArr grayImg;

        for (File image : imageFiles) {

            label = Integer.parseInt(image.getName());

            // Convert image to grayscale:
            img = cvLoadImage(image.getAbsolutePath());
            grayImg = IplImage.create(cvSize(41, 41), IPL_DEPTH_8U, 1);
            cvCvtColor(img, grayImg, CV_BGR2GRAY);

            // Append it in the image list:
            images.put(counter, grayImg);

            // And in the labels list:
//            labels.put(label);

            // Increase counter for next image:
            counter++;
        }


        FaceRecognizerPtr model = createFisherFaceRecognizer(10000);
        //FaceRecognizerPtr faceRecognizer = createEigenFaceRecognizer();
        // FaceRecognizerPtr faceRecognizer = createLBPHFaceRecognizer();

        //model.get().train(images, label);

        // Load the test image:
        IplImage greyTestImage = IplImage.create(cvSize(41, 41), IPL_DEPTH_8U, 1);
        cvCvtColor(testImage, greyTestImage, CV_BGR2GRAY);

        // And get a prediction:
        int predictedLabel = model.get().predict(greyTestImage);
        System.out.println("Predicted label: " + predictedLabel);
    }

    public static String getPathImages(){

        File file = new File(new StringBuilder(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES).getAbsolutePath())
                .append("/")
                .append("SmartApp").toString());

        if (!file.isDirectory())
            file.mkdir();

        return new StringBuilder(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES).getAbsolutePath())
                .append("/")
                .append("SmartApp").toString();
    }

    public static File createFile(Bitmap bitmap){

        //Make sure you set permission Read/Write external storage AND set Provider that exists in Manifest

        String file_path = getPathImages();
        File dir = new File(file_path);
//        if(!dir.exists())
//            dir.mkdirs();
        File file = new File(dir.getAbsoluteFile(), new StringBuilder(UUID.randomUUID().toString()).append(".png").toString());
        FileOutputStream fOut;
        try {
            fOut = new FileOutputStream(file);
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, fOut);
            fOut.flush();
            fOut.close();
        } catch (Exception e) {
            e.printStackTrace();
        }

        return file;
    }

}
