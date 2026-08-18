use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn find_nvcc() -> Option<PathBuf> {
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        let p = PathBuf::from(cuda_path).join("bin").join("nvcc.exe");
        if p.exists() {
            return Some(p);
        }
    }
    let candidates = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin\nvcc.exe",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\nvcc.exe",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin\nvcc.exe",
    ];
    for c in candidates {
        let p = PathBuf::from(c);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

fn find_msvc_bindir() -> Option<PathBuf> {
    let candidates = [
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.43.34808\bin\Hostx64\x64",
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.42.34433\bin\Hostx64\x64",
    ];
    for c in candidates {
        let p = PathBuf::from(c);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

fn main() {
    println!("cargo:rerun-if-changed=src/kernels/kmeans.cu");
    println!("cargo:rerun-if-changed=src/kernels/knn.cu");

    let out_dir = env::var("OUT_DIR").unwrap();
    let out_path = Path::new(&out_dir);

    let nvcc = find_nvcc();
    let msvc = find_msvc_bindir();

    let compile_kernel = |cu_file: &str, ptx_name: &str| {
        let cu_path = format!("src/kernels/{}", cu_file);
        let dst_ptx = out_path.join(ptx_name);

        if let (Some(nvcc_bin), Some(msvc_dir)) = (&nvcc, &msvc) {
            let status = Command::new(nvcc_bin)
                .arg("--compiler-bindir")
                .arg(msvc_dir)
                .arg("--ptx")
                .arg(&cu_path)
                .arg("-o")
                .arg(&dst_ptx)
                .status();

            if let Ok(s) = status
                && s.success()
            {
                return;
            }
        }

        // Fallback to pre-built ptx if compilation was not performed or failed
        let fallback_ptx = format!("src/kernels/{}", ptx_name);
        if Path::new(&fallback_ptx).exists() {
            let _ = fs::copy(&fallback_ptx, &dst_ptx);
        }
    };

    compile_kernel("kmeans.cu", "kmeans.ptx");
    compile_kernel("knn.cu", "knn.ptx");
}
