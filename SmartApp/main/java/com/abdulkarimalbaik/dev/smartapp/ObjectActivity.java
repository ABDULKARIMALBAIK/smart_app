package com.abdulkarimalbaik.dev.smartapp;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Environment;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.SurfaceView;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCamera2View;
import org.opencv.android.JavaCameraView;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class ObjectActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2{


    private static final int PERMISSION_CODE_REQUEST = 1000;
    CameraBridgeViewBase cameraBridgeViewBase;
    BaseLoaderCallback baseLoaderCallback;
    Net tinyYolo;
    int counter = 0;



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_object);

        OpenCVLoader.initDebug();

        if (ActivityCompat.checkSelfPermission(this , Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED
                && ActivityCompat.checkSelfPermission(this , Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED
                && ActivityCompat.checkSelfPermission(this , Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED){

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M)
                requestPermissions(new String[]{Manifest.permission.CAMERA,
                        Manifest.permission.READ_EXTERNAL_STORAGE,
                        Manifest.permission.WRITE_EXTERNAL_STORAGE} , PERMISSION_CODE_REQUEST);
        }
        else
        {
            setupCamera();
        }


    }

    private void setupCamera() {

        cameraBridgeViewBase = (JavaCameraView)findViewById(R.id.CameraView);
        cameraBridgeViewBase.setVisibility(SurfaceView.VISIBLE);
        cameraBridgeViewBase.setCvCameraViewListener(ObjectActivity.this);


        String tinyYoloCfg = Environment.getExternalStorageDirectory() + "/dnns/yolov3-tiny.cfg" ;   //Location in real device
        String tinyYoloWeights = Environment.getExternalStorageDirectory() + "/dnns/yolov3-tiny.weights";  //Location in real device

        tinyYolo = Dnn.readNetFromDarknet(tinyYoloCfg, tinyYoloWeights);


        //System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        baseLoaderCallback = new BaseLoaderCallback(this) {
            @Override
            public void onManagerConnected(int status) {
                super.onManagerConnected(status);

                switch(status){

                    case BaseLoaderCallback.SUCCESS:
                        cameraBridgeViewBase.enableView();
                        break;
                    default:
                        super.onManagerConnected(status);
                        break;
                }


            }

        };
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {

        switch (requestCode){

            case PERMISSION_CODE_REQUEST:{

                if (grantResults[0] == PackageManager.PERMISSION_GRANTED &&
                        grantResults[1] == PackageManager.PERMISSION_GRANTED &&
                        grantResults[2] == PackageManager.PERMISSION_GRANTED){

                    setupCamera();
                }
                else
                    Toast.makeText(this, "You can't use camera , storage !!!", Toast.LENGTH_LONG).show();

                break;
            }
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {

        Toast.makeText(this, "Welcome Smart App", Toast.LENGTH_SHORT).show();
    }

    @Override
    public void onCameraViewStopped() {

    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        Mat frame = inputFrame.rgba();
        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB);
        Mat imageBlob = Dnn.blobFromImage(frame, 0.00392, new Size(416,416),new Scalar(0, 0, 0),/*swapRB*/false, /*crop*/false);
        tinyYolo.setInput(imageBlob);


        java.util.List<Mat> result = new java.util.ArrayList<Mat>(2);
        List<String> outBlobNames = new java.util.ArrayList<>();

        outBlobNames.add(0, "yolo_16");
        outBlobNames.add(1, "yolo_23");

        tinyYolo.forward(result,outBlobNames);

        float confThreshold = 0.3f;
        List<Integer> clsIds = new ArrayList<>();
        List<Float> confs = new ArrayList<>();
        List<Rect> rects = new ArrayList<>();


        for (int i = 0; i < result.size(); ++i)
        {

            Mat level = result.get(i);

            for (int j = 0; j < level.rows(); ++j)
            {
                Mat row = level.row(j);
                Mat scores = row.colRange(5, level.cols());

                Core.MinMaxLocResult mm = Core.minMaxLoc(scores);
                float confidence = (float)mm.maxVal;
                Point classIdPoint = mm.maxLoc;

                if (confidence > confThreshold)
                {
                    int centerX = (int)(row.get(0,0)[0] * frame.cols());
                    int centerY = (int)(row.get(0,1)[0] * frame.rows());
                    int width   = (int)(row.get(0,2)[0] * frame.cols());
                    int height  = (int)(row.get(0,3)[0] * frame.rows());

                    int left    = centerX - width  / 2;
                    int top     = centerY - height / 2;

                    clsIds.add((int)classIdPoint.x);
                    confs.add((float)confidence);

                    rects.add(new Rect(left, top, width, height));
                }
            }
        }

        int ArrayLength = confs.size();

        if (ArrayLength >= 1) {
            // Apply non-maximum suppression procedure.
            float nmsThresh = 0.2f;

            MatOfFloat confidences = new MatOfFloat(Converters.vector_float_to_Mat(confs));


            Rect[] boxesArray = rects.toArray(new Rect[0]);
            MatOfRect boxes = new MatOfRect(boxesArray);
            MatOfInt indices = new MatOfInt();
            Dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThresh, indices);


            // Draw result boxes:
            int[] ind = indices.toArray();
            for (int i = 0; i < ind.length; ++i) {

                int idx = ind[i];
                Rect box = boxesArray[idx];
                int idGuy = clsIds.get(idx);
                float conf = confs.get(idx);

                List<String> cocoNames = Arrays.asList("a person", "a bicycle", "a motorbike", "an airplane", "a bus", "a train", "a truck", "a boat", "a traffic light", "a fire hydrant", "a stop sign", "a parking meter", "a car", "a bench", "a bird", "a cat", "a dog", "a horse", "a sheep", "a cow", "an elephant", "a bear", "a zebra", "a giraffe", "a backpack", "an umbrella", "a handbag", "a tie", "a suitcase", "a frisbee", "skis", "a snowboard", "a sports ball", "a kite", "a baseball bat", "a baseball glove", "a skateboard", "a surfboard", "a tennis racket", "a bottle", "a wine glass", "a cup", "a fork", "a knife", "a spoon", "a bowl", "a banana", "an apple", "a sandwich", "an orange", "broccoli", "a carrot", "a hot dog", "a pizza", "a doughnut", "a cake", "a chair", "a sofa", "a potted plant", "a bed", "a dining table", "a toilet", "a TV monitor", "a laptop", "a computer mouse", "a remote control", "a keyboard", "a cell phone", "a microwave", "an oven", "a toaster", "a sink", "a refrigerator", "a book", "a clock", "a vase", "a pair of scissors", "a teddy bear", "a hair drier", "a toothbrush");
                int intConf = (int) (conf * 100);

                Imgproc.putText(frame,cocoNames.get(idGuy) + " " + intConf + "%",box.tl(),Core.FONT_HERSHEY_SIMPLEX, 2, new Scalar(255,255,0),2);
                Imgproc.rectangle(frame, box.tl(), box.br(), new Scalar(255, 0, 0), 2);
            }
        }



        return frame;
    }

    @Override
    protected void onResume() {
        super.onResume();

        if (!OpenCVLoader.initDebug()){
            Toast.makeText(getApplicationContext(),"There's a problem, yo!", Toast.LENGTH_SHORT).show();
        }
        else
        {
            baseLoaderCallback.onManagerConnected(baseLoaderCallback.SUCCESS);
        }



    }

    @Override
    protected void onPause() {
        super.onPause();

        if(cameraBridgeViewBase!=null){
            cameraBridgeViewBase.disableView();
        }

    }


    @Override
    protected void onDestroy() {
        super.onDestroy();

        if (cameraBridgeViewBase!=null){
            cameraBridgeViewBase.disableView();
        }
    }
}
