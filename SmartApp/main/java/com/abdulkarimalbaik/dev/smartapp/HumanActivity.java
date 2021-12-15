package com.abdulkarimalbaik.dev.smartapp;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Environment;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCamera2View;
import org.opencv.android.JavaCameraView;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;

public class HumanActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2{


    private static final int PERMISSION_CODE_REQUEST = 1000;
    CameraBridgeViewBase cameraBridgeViewBase;
    BaseLoaderCallback baseLoaderCallback;
    Net detector;
    int counter = 0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_human);

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
        cameraBridgeViewBase.setCvCameraViewListener(HumanActivity.this);


        String protoPath = Environment.getExternalStorageDirectory() + "/dnns/deploy.prototxt" ;
        String caffeWeights = Environment.getExternalStorageDirectory() + "/dnns/res10_300x300_ssd_iter_140000.caffemodel";

        detector = Dnn.readNetFromCaffe(protoPath, caffeWeights);


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
        Mat imageBlob = Dnn.blobFromImage(frame, 1.0, new Size(300, 300), new Scalar(104.0, 177.0, 123.0), true, false, CvType.CV_32F);

        detector.setInput(imageBlob); //set the input to network model
        Mat detections = detector.forward(); //feed forward the input to the netwrok to get the output

        int cols = frame.cols();
        int rows = frame.rows();
        double THRESHOLD = 0.55;


        detections = detections.reshape(1, (int)detections.total() / 7);
        Log.d("EXPERIMENT5:ROWS", detections.rows()+"");

        for (int i = 0; i < detections.rows(); ++i) {

            double confidence = detections.get(i, 2)[0];
            Log.d("EXPERIMENT6", i+" "+confidence+" "+THRESHOLD);


            if (confidence > THRESHOLD) {

                int left   = (int)(detections.get(i, 3)[0] * cols);
                int top    = (int)(detections.get(i, 4)[0] * rows);
                int right  = (int)(detections.get(i, 5)[0] * cols);
                int bottom = (int)(detections.get(i, 6)[0] * rows);

                // Draw rectangle around detected object

                if (left<0){
                    left=0;
                }
                if (top<0){
                    top=0;
                }
                if (right<0){
                    right=0;
                }
                if (bottom<0){
                    bottom=0;
                }

                int xLim=frame.size(1);
                int yLim=frame.size(0);

                if (left>=xLim){
                    left=xLim-2;
                }
                if (right>=xLim){
                    right=xLim-2;
                }

                if (top>=yLim){
                    top=yLim-2;
                }
                if (bottom>=yLim){
                    bottom=yLim-2;
                }


                Imgproc.rectangle(frame, new Point(left, top), new Point(right, bottom),new Scalar(255, 255, 0),2);
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
