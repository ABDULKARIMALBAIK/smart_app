package com.abdulkarimalbaik.dev.smartapp;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Build;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.SurfaceView;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCamera2View;
import org.opencv.android.JavaCameraView;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

public class CannyActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2{

    private static final int PERMISSION_CODE_REQUEST = 1000;
    CameraBridgeViewBase cameraBridgeViewBase;
    BaseLoaderCallback baseLoaderCallback;
    int counter = 0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        requestWindowFeature(Window.FEATURE_NO_TITLE); getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,WindowManager.LayoutParams.FLAG_FULLSCREEN);

        setContentView(R.layout.activity_canny);

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
        cameraBridgeViewBase.setCvCameraViewListener(CannyActivity.this);


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

        if (counter % 2 == 0){

            Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2GRAY);
            Imgproc.Canny(frame, frame, 100, 80);
        }

        counter++;

        if (counter == 1000)
            counter = 0;

        return frame;
    }


    @Override
    protected void onResume() {
        super.onResume();

        if (baseLoaderCallback != null){

            if (!OpenCVLoader.initDebug()){
                Toast.makeText(getApplicationContext(),"There's a problem, yo!", Toast.LENGTH_SHORT).show();
            }
            else
            {
                baseLoaderCallback.onManagerConnected(baseLoaderCallback.SUCCESS);
            }
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
