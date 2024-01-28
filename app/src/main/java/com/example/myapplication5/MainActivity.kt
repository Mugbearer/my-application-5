package com.example.myapplication5

import android.graphics.Bitmap
import android.graphics.drawable.BitmapDrawable
import android.media.Image
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.size
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.example.myapplication5.ui.theme.MyApplication5Theme
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import com.example.myapplication5.ml.Model
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MyApplication5Theme {
                // A surface container using the 'background' color from the theme
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    App()
                }
            }
        }
    }
}

@Preview(showBackground = true)
@Composable
fun App() {
    val context = LocalContext.current


    //Gets image from drawables then stores bitmap value
    val drawable = ContextCompat.getDrawable(context, R.drawable.image)
    val bitmap: Bitmap = (drawable as BitmapDrawable).bitmap

    val imageBitmap: ImageBitmap = bitmap.asImageBitmap()


    //Resizes image for model
    var imageProcessor = ImageProcessor.Builder()
        .add(ResizeOp(28,28, ResizeOp.ResizeMethod.BILINEAR))
        .build()


    //Takes bitmap, returns prediction in string
    val useModel: (Bitmap) -> String = {
        var tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(it)

        tensorImage = imageProcessor.process(tensorImage)

        val model = Model.newInstance(context)

        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 28, 28, 3), DataType.FLOAT32)
        inputFeature0.loadBuffer(tensorImage.buffer)

        val outputs = model.process(inputFeature0)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer.floatArray

        var maxIdx = 0

        outputFeature0.forEachIndexed { index, fl ->
            if (outputFeature0[maxIdx] < fl) {
                maxIdx= index
            }
        }

        model.close()

        val classNames = arrayOf("circle","rectangle","square","triangle")

        classNames[maxIdx].toString()
    }

    Column(
        modifier = Modifier
            .fillMaxSize(),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Image(
            bitmap = imageBitmap,
            contentDescription = null,
            modifier = Modifier.size(100.dp)
        )
        Text(
            text = useModel(bitmap)
        )
    }
}

//fun UseModel(bitmap: Bitmap): String  {
//    val model = Model.newInstance(LocalContext.current)
//
//// Creates inputs for reference.
//    val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 28, 28, 3), DataType.FLOAT32)
//    inputFeature0.loadBuffer(byteBuffer)
//
//// Runs model inference and gets result.
//    val outputs = model.process(inputFeature0)
//    val outputFeature0 = outputs.outputFeature0AsTensorBuffer
//
//// Releases model resources if no longer used.
//    model.close()
//
//    return "Hello"
//}