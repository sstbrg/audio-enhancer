plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.audioenhancer.app"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.audioenhancer.app"
        minSdk = 29          // Android 10+ for NNAPI improvements
        targetSdk = 36
        versionCode = 1
        versionName = "0.1.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }

    buildFeatures { viewBinding = true }

    // Don't compress ONNX model in APK
    androidResources {
        noCompress += listOf("onnx", "ort")
    }
}

dependencies {
    // ONNX Runtime — includes NNAPI + XNNPACK execution providers
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.21.0")

    // Audio decoding (FLAC, WAV, MP3, OGG, AAC)
    implementation("androidx.media3:media3-extractor:1.5.1")
    implementation("androidx.media3:media3-exoplayer:1.5.1")
    implementation("androidx.media3:media3-common:1.5.1")

    // USB
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")
}
