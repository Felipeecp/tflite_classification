package com.example.tflite_classification

import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.media.ThumbnailUtils
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import com.example.tflite_classification.ml.Model
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : AppCompatActivity() {

    lateinit var result: TextView
    lateinit var galleryBtn: Button
    lateinit var imageView: ImageView
    lateinit var pictureBtn: Button
    val imageSize = 32

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)


        result = findViewById(R.id.result);
        galleryBtn = findViewById(R.id.galleryBtn)
        imageView = findViewById(R.id.imageView)
        pictureBtn = findViewById(R.id.pictureBtn)


        pictureBtn.setOnClickListener{
            if(checkSelfPermission(android.Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED){
                val cameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
                resultLauncher.launch(cameraIntent)
            }else{
                requestPermissions(arrayOf(android.Manifest.permission.CAMERA),100)
            }
        }
        galleryBtn.setOnClickListener {
            val cameraIntent = Intent(Intent.ACTION_PICK,MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
            resultLauncher.launch(cameraIntent)
        }
    }

    private var resultLauncher = registerForActivityResult(ActivityResultContracts.StartActivityForResult()){
        result ->
        run {
            if (result.resultCode == 1 && result.resultCode == Activity.RESULT_OK) {
                var image: Bitmap = result.data?.extras?.getString("data") as Bitmap
                val dimension = Math.min(image.width, image.height)
                image = ThumbnailUtils.extractThumbnail(image,dimension,dimension)
                imageView.setImageBitmap(image)

                image = Bitmap.createScaledBitmap(image,imageSize,imageSize,false)
                classifyImage(image)
            }else{
                val dat = result.data;
                val image: Bitmap? = null;

            }
        }
    }

    private fun classifyImage(image: Bitmap?) {
        val model = Model.newInstance(applicationContext)

        // Creates inputs for reference.
        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 32, 32, 3), DataType.FLOAT32)
        val byteBuffer = ByteBuffer.allocateDirect(4*imageSize*imageSize*3)
        byteBuffer.order(ByteOrder.nativeOrder())

        val intValues = IntArray(imageSize*imageSize)
        image?.getPixels(intValues, 0, image.width, 0, 0, image.width, image.height)
        var pixel = 0
        for(i in 0..imageSize){
            for (j in 0..imageSize){
                val valor = intValues[pixel++]
                byteBuffer.putFloat(((valor shr 16) and 0xFF) * (1f / 1))
                byteBuffer.putFloat(((valor shr 8) and 0xFF) * (1f / 1))
                byteBuffer.putFloat((valor and 0xFF) * (1f / 1))
            }
        }

        inputFeature0.loadBuffer(byteBuffer)

        // Runs model inference and gets result.
        val outputs = model.process(inputFeature0)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer

        // Releases model resources if no longer used.
        val confidences = outputFeature0.floatArray
        var maxConfidences = 0F;
        var maxPos = 0;
        for(i in 0..confidences.size){
            if(confidences[i] > maxConfidences){
                maxConfidences = confidences[i]
                maxPos = i;
            }
        }
        val classes = arrayOf("apple","banana","orange")
        result.text = classes[maxPos]

        model.close()
    }

}