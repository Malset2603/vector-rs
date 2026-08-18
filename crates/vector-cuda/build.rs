use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn find_nvcc() -> Option<String> {
    let exe_name = if cfg!(windows) { "nvcc.exe" } else { "nvcc" };
    
    if let Ok(path) = env::var("PATH") {
        for dir in env::split_paths(&path) {
            if dir.join(exe_name).exists() {
                return Some(exe_name.to_string());
            }
        }
    }

    if cfg!(target_os = "linux") {
        if Path::new("/usr/local/cuda/bin/nvcc").exists() {
            return Some("/usr/local/cuda/bin/nvcc".to_string());
        }
    } else if cfg!(windows) {
        if let Ok(cuda_path) = env::var("CUDA_PATH") {
            let p = PathBuf::from(cuda_path).join("bin").join("nvcc.exe");
            if p.exists() {
                return Some(p.to_string_lossy().to_string());
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
            if Path::new(c).exists() {
                return Some(c.to_string());
            }
        }
    }
    None
}

fn find_msvc_bindir() -> Option<PathBuf> {
    if !cfg!(windows) {
        return None;
    }
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

        if let Some(nvcc_bin) = &nvcc {
            let mut cmd = Command::new(nvcc_bin);
            
            if cfg!(windows) {
                if let Some(msvc_dir) = &msvc {
                    cmd.arg("--compiler-bindir").arg(msvc_dir);
                }
            }

            let status = cmd.arg("--ptx")
                .arg(&cu_path)
                .arg("-o")
                .arg(&dst_ptx)
                .status();

            if let Ok(s) = status {
                if s.success() {
                    return;
                }
            }
        }

        // Fallback to pre-built ptx if compilation was not performed or failed
        let fallback_ptx = format!("src/kernels/{}", ptx_name);
        if Path::new(&fallback_ptx).exists() {
            let _ = fs::copy(&fallback_ptx, &dst_ptx);
        } else {
            // Force panic so the user knows exactly what went wrong if fallback fails
            if nvcc.is_none() {
                panic!("Failed to find nvcc! Please install CUDA Toolkit and ensure nvcc is in PATH.");
            } else {
                panic!("nvcc failed to compile {}! Check your CUDA installation.", cu_file);
            }
        }
    };

    compile_kernel("kmeans.cu", "kmeans.ptx");
    compile_kernel("knn.cu", "knn.ptx");
}
