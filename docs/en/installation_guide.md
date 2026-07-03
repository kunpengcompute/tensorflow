# Installation Guide

## TensorFlow ANNC for Graph Compilation Optimization

### Verified Environment

To use the TensorFlow ANNC for graph compilation optimization feature smoothly and securely, ensure that your environment is one of the verified environments.

**Hardware Requirements<a name="section230mcpsimp"></a>**

[**Table 1**](#hardware-requirement) describes the verified hardware environment.

**Table 1** Hardware requirement<a id="hardware-requirement"></a>

<table><thead align="left"><tr id="row239mcpsimp"><th class="cellrowborder" valign="top" width="20%" id="mcps1.2.3.1.1"><p id="p241mcpsimp"><a name="p241mcpsimp"></a><a name="p241mcpsimp"></a>Item</p>
</th>
<th class="cellrowborder" valign="top" width="80%" id="mcps1.2.3.1.2"><p id="p243mcpsimp"><a name="p243mcpsimp"></a><a name="p243mcpsimp"></a>Description</p>
</th>
</tr>
</thead>
<tbody><tr id="row245mcpsimp"><td class="cellrowborder" valign="top" width="20%" headers="mcps1.2.3.1.1 "><p id="p247mcpsimp"><a name="p247mcpsimp"></a><a name="p247mcpsimp"></a>CPU</p>
</td>
<td class="cellrowborder" valign="top" width="80%" headers="mcps1.2.3.1.2 "><p id="p249mcpsimp"><a name="p249mcpsimp"></a><a name="p249mcpsimp"></a>New Kunpeng 920 processor model and Kunpeng 950 processor</p>
</td>
</tr>
</tbody>
</table>

**OS Requirements<a name="section250mcpsimp"></a>**

[**Table 2**](#os-requirements) lists the verified OSs.

**Table 2** OS requirements<a id="os-requirements"></a>

<a name="_d0e164"></a>

<table><thead align="left"><tr id="row261mcpsimp"><th class="cellrowborder" valign="top" width="10.24%" id="mcps1.2.5.1.1"><p id="p263mcpsimp"><a name="p263mcpsimp"></a><a name="p263mcpsimp"></a>Item</p>
</th>
<th class="cellrowborder" valign="top" width="23.73%" id="mcps1.2.5.1.2"><p id="p265mcpsimp"><a name="p265mcpsimp"></a><a name="p265mcpsimp"></a>Version</p>
</th>
<th class="cellrowborder" valign="top" width="45.519999999999996%" id="mcps1.2.5.1.3"><p id="p267mcpsimp"><a name="p267mcpsimp"></a><a name="p267mcpsimp"></a>Description</p>
</th>
<th class="cellrowborder" valign="top" width="20.51%" id="mcps1.2.5.1.4"><p id="p269mcpsimp"><a name="p269mcpsimp"></a><a name="p269mcpsimp"></a>How to Obtain</p>
</th>
</tr>
</thead>
<tbody><tr id="row271mcpsimp"><td class="cellrowborder" valign="top" width="10.24%" headers="mcps1.2.5.1.1 "><p id="p273mcpsimp"><a name="p273mcpsimp"></a><a name="p273mcpsimp"></a>OS</p>
</td>
<td class="cellrowborder" valign="top" width="23.73%" headers="mcps1.2.5.1.2 "><p id="p275mcpsimp"><a name="p275mcpsimp"></a><a name="p275mcpsimp"></a>openEuler 22.03 LTS SP3</p>
</td>
<td class="cellrowborder" valign="top" width="45.519999999999996%" headers="mcps1.2.5.1.3 "><p id="p277mcpsimp"><a name="p277mcpsimp"></a><a name="p277mcpsimp"></a>When installing an OS, choose <code>Minimal Install</code> and select <code>Development Tools</code> to minimize manual operations.</p>
</td>
<td class="cellrowborder" valign="top" width="20.51%" headers="mcps1.2.5.1.4 "><p id="p279mcpsimp"><a name="p279mcpsimp"></a><a name="p279mcpsimp"></a><a href="https://repo.openeuler.org/openEuler-22.03-LTS-SP3/ISO/aarch64/" target="_blank" rel="noopener noreferrer">Link</a></p>
</td>
</tr>
<tr id="row281mcpsimp"><td class="cellrowborder" valign="top" width="10.24%" headers="mcps1.2.5.1.1 "><p id="p283mcpsimp"><a name="p283mcpsimp"></a><a name="p283mcpsimp"></a>Kernel</p>
</td>
<td class="cellrowborder" valign="top" width="23.73%" headers="mcps1.2.5.1.2 "><p id="p285mcpsimp"><a name="p285mcpsimp"></a><a name="p285mcpsimp"></a>5.10.0</p>
</td>
<td class="cellrowborder" valign="top" width="45.519999999999996%" headers="mcps1.2.5.1.3 "><p id="p287mcpsimp"><a name="p287mcpsimp"></a><a name="p287mcpsimp"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="20.51%" headers="mcps1.2.5.1.4 "><p id="p289mcpsimp"><a name="p289mcpsimp"></a><a name="p289mcpsimp"></a>Included in the OS image</p>
</td>
</tr>
</tbody>
</table>

**Software Requirements<a name="section290mcpsimp"></a>**

[**Table 3**](#software-requirements) describes the verified software dependencies.

**Table 3** Software requirements<a id="software-requirements"></a>

<a name="_table237115053311"></a>

<table><thead align="left"><tr id="row301mcpsimp"><th class="cellrowborder" valign="top" width="10.100000000000001%" id="mcps1.2.5.1.1"><p id="p303mcpsimp"><a name="p303mcpsimp"></a><a name="p303mcpsimp"></a>Item</p>
</th>
<th class="cellrowborder" valign="top" width="23.84%" id="mcps1.2.5.1.2"><p id="p305mcpsimp"><a name="p305mcpsimp"></a><a name="p305mcpsimp"></a>Version</p>
</th>
<th class="cellrowborder" valign="top" width="45.48%" id="mcps1.2.5.1.3"><p id="p307mcpsimp"><a name="p307mcpsimp"></a><a name="p307mcpsimp"></a>Description</p>
</th>
<th class="cellrowborder" valign="top" width="20.580000000000002%" id="mcps1.2.5.1.4"><p id="p309mcpsimp"><a name="p309mcpsimp"></a><a name="p309mcpsimp"></a>How to Obtain</p>
</th>
</tr>
</thead>
<tbody><tr id="row321mcpsimp"><td class="cellrowborder" valign="top" width="10.100000000000001%" headers="mcps1.2.5.1.1 "><p id="p323mcpsimp"><a name="p323mcpsimp"></a><a name="p323mcpsimp"></a>Python</p>
</td>
<td class="cellrowborder" valign="top" width="23.84%" headers="mcps1.2.5.1.2 "><p id="p325mcpsimp"><a name="p325mcpsimp"></a><a name="p325mcpsimp"></a>3.9.9</p>
</td>
<td class="cellrowborder" valign="top" width="45.48%" headers="mcps1.2.5.1.3 "><p id="p327mcpsimp"><a name="p327mcpsimp"></a><a name="p327mcpsimp"></a>Python is an auxiliary tool used during the build of TF Serving. It is used to automatically download dependencies and configure the environment.</p>
</td>
<td class="cellrowborder" valign="top" width="20.580000000000002%" headers="mcps1.2.5.1.4 "><p id="p329mcpsimp"><a name="p329mcpsimp"></a><a name="p329mcpsimp"></a>Install it using a Yum repository.</p>
</td>
</tr>
<tr id="row330mcpsimp"><td class="cellrowborder" valign="top" width="10.100000000000001%" headers="mcps1.2.5.1.1 "><p id="p332mcpsimp"><a name="p332mcpsimp"></a><a name="p332mcpsimp"></a>CMake</p>
</td>
<td class="cellrowborder" valign="top" width="23.84%" headers="mcps1.2.5.1.2 "><p id="p334mcpsimp"><a name="p334mcpsimp"></a><a name="p334mcpsimp"></a>3.22.0</p>
</td>
<td class="cellrowborder" valign="top" width="45.48%" headers="mcps1.2.5.1.3 "><p id="p336mcpsimp"><a name="p336mcpsimp"></a><a name="p336mcpsimp"></a>CMake is the build tool of TF Serving. The CMake version must be 3.22.0 or later.</p>
</td>
<td class="cellrowborder" valign="top" width="20.580000000000002%" headers="mcps1.2.5.1.4 "><p id="p338mcpsimp"><a name="p338mcpsimp"></a><a name="p338mcpsimp"></a>Install it using a Yum repository.</p>
</td>
</tr>
<tr id="row339mcpsimp"><td class="cellrowborder" valign="top" width="10.100000000000001%" headers="mcps1.2.5.1.1 "><p id="p341mcpsimp"><a name="p341mcpsimp"></a><a name="p341mcpsimp"></a>GCC/G++</p>
</td>
<td class="cellrowborder" valign="top" width="23.84%" headers="mcps1.2.5.1.2 "><p id="p343mcpsimp"><a name="p343mcpsimp"></a><a name="p343mcpsimp"></a>12.3.1</p>
</td>
<td class="cellrowborder" valign="top" width="45.48%" headers="mcps1.2.5.1.3 "><p id="p345mcpsimp"><a name="p345mcpsimp"></a><a name="p345mcpsimp"></a>GNU Compiler Collection (GCC) serves as the compiler for compiling TF Serving into an executable file.</p>
</td>
<td class="cellrowborder" valign="top" width="20.580000000000002%" headers="mcps1.2.5.1.4 "><p id="p347mcpsimp"><a name="p347mcpsimp"></a><a name="p347mcpsimp"></a>Install it using a Yum repository.</p>
</td>
</tr>
<tr id="row348mcpsimp"><td class="cellrowborder" valign="top" width="10.100000000000001%" headers="mcps1.2.5.1.1 "><p id="p8254102818211"><a name="p8254102818211"></a><a name="p8254102818211"></a>Bazel</p>
</td>
<td class="cellrowborder" valign="top" width="23.84%" headers="mcps1.2.5.1.2 "><p id="p525314288218"><a name="p525314288218"></a><a name="p525314288218"></a>6.5.0</p>
</td>
<td class="cellrowborder" valign="top" width="45.48%" headers="mcps1.2.5.1.3 "><p id="p10253172810219"><a name="p10253172810219"></a><a name="p10253172810219"></a>Bazel is a powerful build system that enables fast and scalable building. The Bazel version must be 6.4.0 or later.</p>
</td>
<td class="cellrowborder" valign="top" width="20.51%" headers="mcps1.2.5.1.4 "><p id="p279mcpsimp"><a name="p279mcpsimp"></a><a name="p279mcpsimp"></a><a href="https://releases.bazel.build/6.5.0/release/bazel-6.5.0-dist.zip" target="_blank" rel="noopener noreferrer">Link</a></p>
</td>
</tr>
<tr id="row8837155216431"><td class="cellrowborder" valign="top" width="10.100000000000001%" headers="mcps1.2.5.1.1 "><p id="p17838135224315"><a name="p17838135224315"></a><a name="p17838135224315"></a>TensorFlow</p>
</td>
<td class="cellrowborder" valign="top" width="23.84%" headers="mcps1.2.5.1.2 "><p id="p138385526435"><a name="p138385526435"></a><a name="p138385526435"></a>2.15.1</p>
</td>
<td class="cellrowborder" valign="top" width="45.48%" headers="mcps1.2.5.1.3 "><p id="p7838052104314"><a name="p7838052104314"></a><a name="p7838052104314"></a>TensorFlow, developed by Google, is an open-source machine learning framework. It supports end-to-end AI model development and deployment from research to production.</p>
</td>
<td class="cellrowborder" valign="top" width="20.580000000000002%" headers="mcps1.2.5.1.4 "><p id="p10838352174310"><a name="p10838352174310"></a><a name="p10838352174310"></a>Install it using a pip repository.</p>
</td>
</tr>
</tbody>
</table>

### Setting Up the Environment

Prepare the TF Serving compilation environment by following instructions in [Configuring the Compilation Environment](https://www.hikunpeng.com/document/detail/en/SRA/ecosystemEnable/TensorFlowServing/kunpengtfserving_06_0005.html) in the _TensorFlow Serving Porting Guide_ and [**Table 3**](#software-requirements).

### Compilation and Installation<a name="ZH-CN_TOPIC_0000002549929907"></a>

The backend implementation of TensorFlow ANNC for graph compilation optimization has been merged into the TensorFlow and TF Serving open-source repositories. Code related to the Tensorflow graph fusion, XLA graph fusion, operator optimization, and constant folding optimization are available in the ANNC open-source repository hosted on Gitcode, which can be cloned and compiled using `git`.

**Obtaining Code<a name="section12640151210390"></a>**

1. Disable SSL verification for Git.

   ```bash
   git config --global http.sslVerify false
   git config --global https.sslVerify false
   ```

2. Obtain the TensorFlow and TF Serving code.

   ```bash
   git clone --branch v2.15.0-2509 https://gitcode.com/boostkit/tensorflow.git
   git clone --branch v2.15.1-2509 https://gitcode.com/boostkit/tensorflow-serving.git
   ```

3. Obtain ANNC code.

   ```bash
   git clone --branch v0.0.4 https://gitcode.com/openeuler/ANNC.git
   ```

**Performing Compilation and Installation<a name="section176411312143912"></a>**

1. Install GCC 12.3.1.

   ```bash
   yum install -y gcc-toolset-12-gcc*
   export PATH=/opt/openEuler/gcc-toolset-12/root/usr/bin/:$PATH
   export LD_LIBRARY_PATH=/opt/openEuler/gcc-toolset-12/root/usr/lib64/:$LD_LIBRARY_PATH
   ```

2. Compile and install ANNC.

   ```bash
   export ANNC=/path/to/ANNC
   cd $ANNC
   source build.sh
   cp bazel-bin/annc/service/cpu/libannc.so /usr/lib64/
   cp $ANNC/annc/service/cpu/xla/libs/XNNPACK/build/libXNNPACK.so /usr/lib64
   cp $ANNC/bazel-out/aarch64-opt/bin/annc/service/cpu/xla/libs/libblas_mlir.so /usr/lib64
   mkdir -p /usr/include/annc 
   cp annc/service/cpu/kdnn_rewriter.h /usr/include/annc/ 
   cp annc/service/cpu/annc_flags.h /usr/include/annc/
   cd python
   python3 setup.py bdist_wheel
   python3 -m pip install dist/*.whl --force-reinstall
   ```

3. Enable the patch.

   ```bash
   export TF_PATH=/path/to/tensorflow
   export XLA_PATH=/path/to/tensorflow/third_party/xla
   cd $ANNC/install/tfserver/xla
   bash xla2.sh
   ```

   [**Figure 1**](#successful-patching) shows an example of successful patching.

   **Figure 1** Successful patching<a name="fig5303357205213"></a><a id="successful-patching"></a>

   ![successful-patching](figures/successful-patching.png "Successful patching")
4. Go to the `tensorflow-serving` directory.

   ```bash
   cd /path/to/tensorflow-serving/
   ```

5. Create a directory for storing compilation dependencies.

   ```bash
   export DISTDIR=$(pwd)/download
   mkdir -p $DISTDIR
   ```

6. Set the `BAZEL_PATH` to the directory that contains the `bazel` executable file.

   ```bash
   export BAZEL_PATH=/path/to/bazel
   ```

7. Run the build script to compile the code.

   ```bash
   sh compile_serving.sh --tensorflow_dir /path/to/tensorflow --features gcc12,annc
   ```

   `/path/to/tensorflow` specifies the TensorFlow path.

   The build result is the TF Serving binary file `tensorflow_model_server`, and the file path is `/path/to/tensorflow-serving/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server`.

   In the preceding command, `gcc12` indicates that GCC 12.3.1 is used for compilation. The compile command in the build script is as follows:

   ```bash
   bazel --output_user_root=$BAZEL_COMPILE_CACHE build -c opt --distdir=$DISTDIR --override_repository=org_tensorflow=$TENSORFLOW_DIR \
   --copt=-march=armv8.3-a+crc --copt=-O3 --copt=-fprefetch-loop-arrays --copt=-Wno-error=maybe-uninitialized  \ 
   --copt=-Werror=stringop-overflow=0 --config=fused_embedding  \ 
   --define tflite_with_xnnpack=false tensorflow_serving/model_servers:tensorflow_model_server
   ```

   Some of the parameters are as follows:
   - `--output_user_root`: Bazel compilation cache directory. The default value is `/path/to/tensorflow-serving/output`. You can set a custom path using the `BAZEL_COMPILE_CACHE` environment variable. The command is as follows:

     ```bash
     export BAZEL_COMPILE_CACHE=/path/to/your/cache_dir
     ```

   - `--distdir`: directory for storing TF Serving compilation dependencies. It ensures reliable access when third-party package download fails.
   - `--override_repository`: specifies the local TensorFlow build and uses the directory specified by `tensorflow_dir` as the local TensorFlow.

## TensorFlow Serving Thread Scheduling Optimization

### Verified Environment

**Hardware Requirements<a name="section230mcpsimp"></a>**

[**Table 1**](#hardware-requirement-1) describes the verified hardware environment.

**Table 1** Hardware requirement<a id="hardware-requirement-1"></a>

<a name="_table38928044"></a>

<table><thead align="left"><tr id="row239mcpsimp"><th class="cellrowborder" valign="top" width="25%" id="mcps1.2.3.1.1"><p id="p241mcpsimp"><a name="p241mcpsimp"></a><a name="p241mcpsimp"></a>Item</p>
</th>
<th class="cellrowborder" valign="top" width="75%" id="mcps1.2.3.1.2"><p id="p243mcpsimp"><a name="p243mcpsimp"></a><a name="p243mcpsimp"></a>Description</p>
</th>
</tr>
</thead>
<tbody><tr id="row245mcpsimp"><td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.3.1.1 "><p id="p247mcpsimp"><a name="p247mcpsimp"></a><a name="p247mcpsimp"></a>CPU</p>
</td>
<td class="cellrowborder" valign="top" width="75%" headers="mcps1.2.3.1.2 "><p id="p249mcpsimp"><a name="p249mcpsimp"></a><a name="p249mcpsimp"></a>New Kunpeng 920 processor model (80 cores)</p>
</td>
</tr>
</tbody>
</table>

**OS Requirements<a name="section250mcpsimp"></a>**

[**Table 2**](#os-requirements-1) lists the verified OSs.

**Table 2** OS requirements<a id="os-requirements-1"></a>

<a name="_d0e164"></a>

<table><thead align="left"><tr id="row261mcpsimp"><th class="cellrowborder" valign="top" width="15%" id="mcps1.2.5.1.1"><p id="p263mcpsimp"><a name="p263mcpsimp"></a><a name="p263mcpsimp"></a>Item</p>
</th>
<th class="cellrowborder" valign="top" width="16%" id="mcps1.2.5.1.2"><p id="p265mcpsimp"><a name="p265mcpsimp"></a><a name="p265mcpsimp"></a>Version</p>
</th>
<th class="cellrowborder" valign="top" width="35%" id="mcps1.2.5.1.3"><p id="p267mcpsimp"><a name="p267mcpsimp"></a><a name="p267mcpsimp"></a>Description</p>
</th>
<th class="cellrowborder" valign="top" width="34%" id="mcps1.2.5.1.4"><p id="p269mcpsimp"><a name="p269mcpsimp"></a><a name="p269mcpsimp"></a>How to Obtain</p>
</th>
</tr>
</thead>
<tbody><tr id="row271mcpsimp"><td class="cellrowborder" valign="top" width="15%" headers="mcps1.2.5.1.1 "><p id="p273mcpsimp"><a name="p273mcpsimp"></a><a name="p273mcpsimp"></a>OS</p>
</td>
<td class="cellrowborder" valign="top" width="16%" headers="mcps1.2.5.1.2 "><p id="p275mcpsimp"><a name="p275mcpsimp"></a><a name="p275mcpsimp"></a>openEuler 22.03 LTS SP3</p>
</td>
<td class="cellrowborder" valign="top" width="35%" headers="mcps1.2.5.1.3 "><p id="p277mcpsimp"><a name="p277mcpsimp"></a><a name="p277mcpsimp"></a>When installing an OS, choose <code>Minimal Install</code> and select <code>Development Tools</code> to minimize manual operations.</p>
</td>
<td class="cellrowborder" valign="top" width="34%" headers="mcps1.2.5.1.4 "><p id="p279mcpsimp"><a name="p279mcpsimp"></a><a name="p279mcpsimp"></a><a href="https://repo.openeuler.org/openEuler-22.03-LTS-SP3/ISO/aarch64/" target="_blank" rel="noopener noreferrer">Link</a></p>
</td>
</tr>
<tr id="row281mcpsimp"><td class="cellrowborder" valign="top" width="15%" headers="mcps1.2.5.1.1 "><p id="p283mcpsimp"><a name="p283mcpsimp"></a><a name="p283mcpsimp"></a>Kernel</p>
</td>
<td class="cellrowborder" valign="top" width="16%" headers="mcps1.2.5.1.2 "><p id="p285mcpsimp"><a name="p285mcpsimp"></a><a name="p285mcpsimp"></a>5.10.0</p>
</td>
<td class="cellrowborder" valign="top" width="35%" headers="mcps1.2.5.1.3 "><p id="p287mcpsimp"><a name="p287mcpsimp"></a><a name="p287mcpsimp"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="34%" headers="mcps1.2.5.1.4 "><p id="p289mcpsimp"><a name="p289mcpsimp"></a><a name="p289mcpsimp"></a>Included in the OS image</p>
</td>
</tr>
</tbody>
</table>

**Software Requirements<a name="section290mcpsimp"></a>**

[**Table 3**](#software-requirements-1) describes the verified software dependencies.

**Table 3** Software requirements<a id="software-requirements-1"></a>

<a name="_table237115053311"></a>

<table><thead align="left"><tr id="row301mcpsimp"><th class="cellrowborder" valign="top" width="15%" id="mcps1.2.5.1.1"><p id="p303mcpsimp"><a name="p303mcpsimp"></a><a name="p303mcpsimp"></a>Item</p>
</th>
<th class="cellrowborder" valign="top" width="15.98%" id="mcps1.2.5.1.2"><p id="p305mcpsimp"><a name="p305mcpsimp"></a><a name="p305mcpsimp"></a>Version</p>
</th>
<th class="cellrowborder" valign="top" width="35.02%" id="mcps1.2.5.1.3"><p id="p307mcpsimp"><a name="p307mcpsimp"></a><a name="p307mcpsimp"></a>Description</p>
</th>
<th class="cellrowborder" valign="top" width="34%" id="mcps1.2.5.1.4"><p id="p309mcpsimp"><a name="p309mcpsimp"></a><a name="p309mcpsimp"></a>How to Obtain</p>
</th>
</tr>
</thead>
<tbody><tr id="row321mcpsimp"><td class="cellrowborder" valign="top" width="15%" headers="mcps1.2.5.1.1 "><p id="p323mcpsimp"><a name="p323mcpsimp"></a><a name="p323mcpsimp"></a>Python</p>
</td>
<td class="cellrowborder" valign="top" width="15.98%" headers="mcps1.2.5.1.2 "><p id="p325mcpsimp"><a name="p325mcpsimp"></a><a name="p325mcpsimp"></a>3.9.9</p>
</td>
<td class="cellrowborder" valign="top" width="35.02%" headers="mcps1.2.5.1.3 "><p id="p327mcpsimp"><a name="p327mcpsimp"></a><a name="p327mcpsimp"></a>Python is an auxiliary tool used during the build of TF Serving. It is used to automatically download dependencies and configure the environment.</p>
</td>
<td class="cellrowborder" valign="top" width="34%" headers="mcps1.2.5.1.4 "><p id="p329mcpsimp"><a name="p329mcpsimp"></a><a name="p329mcpsimp"></a>Install it using a Yum repository.</p>
</td>
</tr>
<tr id="row330mcpsimp"><td class="cellrowborder" valign="top" width="15%" headers="mcps1.2.5.1.1 "><p id="p332mcpsimp"><a name="p332mcpsimp"></a><a name="p332mcpsimp"></a>CMake</p>
</td>
<td class="cellrowborder" valign="top" width="15.98%" headers="mcps1.2.5.1.2 "><p id="p334mcpsimp"><a name="p334mcpsimp"></a><a name="p334mcpsimp"></a>3.22.0</p>
</td>
<td class="cellrowborder" valign="top" width="35.02%" headers="mcps1.2.5.1.3 "><p id="p336mcpsimp"><a name="p336mcpsimp"></a><a name="p336mcpsimp"></a>CMake is the build tool of TF Serving. The CMake version must be 3.22.0 or later.</p>
</td>
<td class="cellrowborder" valign="top" width="34%" headers="mcps1.2.5.1.4 "><p id="p338mcpsimp"><a name="p338mcpsimp"></a><a name="p338mcpsimp"></a>Install it using a Yum repository.</p>
</td>
</tr>
<tr id="row339mcpsimp"><td class="cellrowborder" valign="top" width="15%" headers="mcps1.2.5.1.1 "><p id="p341mcpsimp"><a name="p341mcpsimp"></a><a name="p341mcpsimp"></a>GCC/G++</p>
</td>
<td class="cellrowborder" valign="top" width="15.98%" headers="mcps1.2.5.1.2 "><p id="p343mcpsimp"><a name="p343mcpsimp"></a><a name="p343mcpsimp"></a>12.3.1</p>
</td>
<td class="cellrowborder" valign="top" width="35.02%" headers="mcps1.2.5.1.3 "><p id="p345mcpsimp"><a name="p345mcpsimp"></a><a name="p345mcpsimp"></a>GNU Compiler Collection (GCC) serves as the compiler for compiling TF Serving into an executable file.</p>
</td>
<td class="cellrowborder" valign="top" width="34%" headers="mcps1.2.5.1.4 "><p id="p347mcpsimp"><a name="p347mcpsimp"></a><a name="p347mcpsimp"></a>Install it using a Yum repository.</p>
</td>
</tr>
<tr id="row348mcpsimp"><td class="cellrowborder" valign="top" width="15%" headers="mcps1.2.5.1.1 "><p id="p8254102818211"><a name="p8254102818211"></a><a name="p8254102818211"></a>Bazel</p>
</td>
<td class="cellrowborder" valign="top" width="15.98%" headers="mcps1.2.5.1.2 "><p id="p525314288218"><a name="p525314288218"></a><a name="p525314288218"></a>6.5.0</p>
</td>
<td class="cellrowborder" valign="top" width="35.02%" headers="mcps1.2.5.1.3 "><p id="p10253172810219"><a name="p10253172810219"></a><a name="p10253172810219"></a>Bazel is a powerful build system that enables fast and scalable building. The Bazel version must be 6.4.0 or later.</p>
</td>
<td class="cellrowborder" valign="top" width="20.51%" headers="mcps1.2.5.1.4 "><p id="p279mcpsimp"><a name="p279mcpsimp"></a><a name="p279mcpsimp"></a><a href="https://releases.bazel.build/6.5.0/release/bazel-6.5.0-dist.zip" target="_blank" rel="noopener noreferrer">Link</a></p>
</td>
</tr>
</tbody>
</table>

### Setting Up the Environment

Prepare the TF Serving compilation environment by following instructions in [Configuring the Compilation Environment](https://www.hikunpeng.com/document/detail/en/SRA/ecosystemEnable/TensorFlowServing/kunpengtfserving_06_0005.html) in the _TensorFlow Serving Porting Guide_.

### Compilation and Installation

The patches for TF Serving thread scheduling optimization are now available in the open-source TensorFlow and TF Serving repositories hosted on Gitee. You can clone these repositories via Git and proceed with compilation.

**Obtaining Code<a name="section834119269284"></a>**

1. Disable SSL verification for Git.

   ```bash
   git config --global http.sslVerify false
   git config --global https.sslVerify false
   ```

2. Pull the code.

   ```bash
   git clone https://gitee.com/openeuler/sra_tensorflow_adapter.git -b v2.15.1.0
   ```

**Performing Compilation<a name="section934216265289"></a>**

1. Install GCC 12.3.1.

   ```bash
   yum install -y gcc-toolset-12-gcc*
   PATH=/opt/openEuler/gcc-toolset-12/root/usr/bin/:$PATH
   LD_LIBRARY_PATH=/opt/openEuler/gcc-toolset-12/root/usr/lib64/:$LD_LIBRARY_PATH
   ```

2. Go to the `sra_tensorflow_adapter` directory.

   ```bash
   cd sra_tensorflow_adapter/
   ```

3. Create a directory for storing compilation dependencies. The path is `/path/to/sra_tensorflow_adapter/serving/download`.

   ```bash
   export DISTDIR=$(pwd)/serving/download
   mkdir -p $DISTDIR
   ```

4. Set the `BAZEL_PATH` to the directory that contains the `bazel` executable file.

   ```bash
   export BAZEL_PATH=/path/to/bazel
   ```

5. Run the build script to compile the code.

   ```bash
   sh compile_serving.sh gcc12
   ```

   The build result is the TF Serving binary file `tensorflow_model_server`, and the file path is `/path/to/sra_tensorflow_adapter/serving/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server`.

   In the preceding command, `gcc12` indicates that GCC 12.3.1 is used for compilation. The compile command in the build script is as follows:

   ```bash
   bazel --output_user_root=$BAZEL_COMPILE_CACHE build -c opt --distdir=$DISTDIR --override_repository=org_tensorflow=$TENSORFLOW_DIR --copt=-march=armv8.3-a+crc --copt=-O3 --copt=-fprefetch-loop-arrays --copt=-Wno-error=maybe-uninitialized --copt=-Werror=stringop-overflow=0 tensorflow_serving/model_servers:tensorflow_model_server
   ```

   Some of the parameters are as follows:
   - `output_user_root`: Bazel compilation cache directory. The default value is `/path/to/sra_tensorflow_adapter/serving/output`. You can set a custom path using the `BAZEL_COMPILE_CACHE` environment variable. The command is as follows:

     ```bash
     export BAZEL_COMPILE_CACHE=/path/to/your/cache_dir
     ```

   - `distdir`: directory for storing TF Serving compilation dependencies. It ensures reliable access when third-party package download fails.
   - `override_repository`: specifies the local TensorFlow build. The `/path/to/sra_tensorflow_adapter/tensorflow` directory is automatically identified as the build dependency.

## TensorFlow–KDNN Integration

Kunpeng Deep Neural Network Library (KDNN) is a high-performance AI operator library optimized for the Kunpeng platform. These optimizations are delivered by integrating operators such as MatMul, FusedMatMul, and SparseMatmul into TensorFlow. Integrating KDNN can reduce the latency of Neural Network (NN) operators and greatly improve the model inference performance.

1. Obtain the [KDNN software package](https://gitcode.com/boostkit/boostsra/releases/download/v1.1.0/BoostKit-boostcore-kdnn_3.0.0.zip) for GCC. Decompress the ZIP file to obtain the RPM installation package.
2. Install KDNN.

   ```bash
   rpm -ivh boostcore-kdnn-3.0.0-1.aarch64.rpm
   ```

   The header file installation directory is `/usr/local/kdnn/include`, and the library file installation directories are `/usr/local/kdnn/lib/threadpool` and `/usr/local/kdnn/lib/omp`.
3. Install KDNN header files and static libraries to the `/path/to/tensorflow/third_party/KDNN` directory.

   ```bash
   export TF_PATH=/path/to/tensorflow
   mkdir -p $TF_PATH/third_party/KDNN/src
   cp -r /usr/local/kdnn/include $TF_PATH/third_party/KDNN
   cp -r /usr/local/kdnn/lib/threadpool/libkdnn.a $TF_PATH/third_party/KDNN/src
   ```

4. Go to the KDNN directory and apply the header file patch to fix TensorFlow's exception handling limitation.

   ```bash
   cd $TF_PATH/third_party/KDNN
   patch -p0 < $TF_PATH/third_party/KDNN/tensorflow_kdnn_include_adapter.patch
   ```

5. Run the build script to compile the code.

   ```bash
   cd /path/to/serving
   sh compile_serving.sh --tensorflow_dir /path/to/tensorflow --features gcc12,kdnn
   ```

## TensorFlow KDNN Thread Passthrough

### Verified Environment

**Hardware Requirements<a name="section230mcpsimp"></a>**

[**Table 1** Hardware requirement](#hardware-requirement-2) describes the verified hardware environment.

**Table 1** Hardware requirement<a id="hardware-requirement-2"></a>

<table><thead align="left"><tr id="row239mcpsimp"><th class="cellrowborder" valign="top" width="25%" id="mcps1.2.3.1.1"><p id="p241mcpsimp"><a name="p241mcpsimp"></a><a name="p241mcpsimp"></a>Item</p>
</th>
<th class="cellrowborder" valign="top" width="75%" id="mcps1.2.3.1.2"><p id="p243mcpsimp"><a name="p243mcpsimp"></a><a name="p243mcpsimp"></a>Description</p>
</th>
</tr>
</thead>
<tbody><tr id="row245mcpsimp"><td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.3.1.1 "><p id="p247mcpsimp"><a name="p247mcpsimp"></a><a name="p247mcpsimp"></a>CPU</p>
</td>
<td class="cellrowborder" valign="top" width="75%" headers="mcps1.2.3.1.2 "><p id="p249mcpsimp"><a name="p249mcpsimp"></a><a name="p249mcpsimp"></a>New Kunpeng 920 processor model (80 cores)</p>
</td>
</tr>
</tbody>
</table>

**OS Requirements<a name="section250mcpsimp"></a>**

[**Table 2** OS requirements](#os-requirements-2) lists the verified OSs.

**Table 2** OS requirements<a id="os-requirements-2"></a>

<table><thead align="left"><tr id="row261mcpsimp"><th class="cellrowborder" valign="top" width="15%" id="mcps1.2.5.1.1"><p id="p263mcpsimp"><a name="p263mcpsimp"></a><a name="p263mcpsimp"></a>Item</p>
</th>
<th class="cellrowborder" valign="top" width="16%" id="mcps1.2.5.1.2"><p id="p265mcpsimp"><a name="p265mcpsimp"></a><a name="p265mcpsimp"></a>Version</p>
</th>
<th class="cellrowborder" valign="top" width="35%" id="mcps1.2.5.1.3"><p id="p267mcpsimp"><a name="p267mcpsimp"></a><a name="p267mcpsimp"></a>Description</p>
</th>
<th class="cellrowborder" valign="top" width="34%" id="mcps1.2.5.1.4"><p id="p269mcpsimp"><a name="p269mcpsimp"></a><a name="p269mcpsimp"></a>How to Obtain</p>
</th>
</tr>
</thead>
<tbody><tr id="row271mcpsimp"><td class="cellrowborder" valign="top" width="15%" headers="mcps1.2.5.1.1 "><p id="p273mcpsimp"><a name="p273mcpsimp"></a><a name="p273mcpsimp"></a>OS</p>
</td>
<td class="cellrowborder" valign="top" width="16%" headers="mcps1.2.5.1.2 "><p id="p275mcpsimp"><a name="p275mcpsimp"></a><a name="p275mcpsimp"></a>openEuler 24.03 LTS SP3</p>
</td>
<td class="cellrowborder" valign="top" width="35%" headers="mcps1.2.5.1.3 "><p id="p277mcpsimp"><a name="p277mcpsimp"></a><a name="p277mcpsimp"></a>When installing an OS, choose <code>Minimal Install</code> and select <code>Development Tools</code> to minimize manual operations.</p>
</td>
<td class="cellrowborder" valign="top" width="34%" headers="mcps1.2.5.1.4 "><p id="p279mcpsimp"><a name="p279mcpsimp"></a><a name="p279mcpsimp"></a><a href="https://repo.openeuler.org/openEuler-24.03-LTS-SP3/ISO/aarch64/" target="_blank" rel="noopener noreferrer">Link</a></p>
</td>
</tr>
<tr id="row281mcpsimp"><td class="cellrowborder" valign="top" width="15%" headers="mcps1.2.5.1.1 "><p id="p283mcpsimp"><a name="p283mcpsimp"></a><a name="p283mcpsimp"></a>Kernel</p>
</td>
<td class="cellrowborder" valign="top" width="16%" headers="mcps1.2.5.1.2 "><p id="p285mcpsimp"><a name="p285mcpsimp"></a><a name="p285mcpsimp"></a>6.6.0</p>
</td>
<td class="cellrowborder" valign="top" width="35%" headers="mcps1.2.5.1.3 "><p id="p287mcpsimp"><a name="p287mcpsimp"></a><a name="p287mcpsimp"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="34%" headers="mcps1.2.5.1.4 "><p id="p289mcpsimp"><a name="p289mcpsimp"></a><a name="p289mcpsimp"></a>Included in the OS image</p>
</td>
</tr>
</tbody>
</table>

**Software Requirements<a name="section290mcpsimp"></a>**

[**Table 3** Software requirements](#software-requirements-2) describes the verified software environments.

**Table 3** Software requirements<a id="software-requirements-2"></a>

<table><thead align="left"><tr id="row301mcpsimp"><th class="cellrowborder" valign="top" width="15%" id="mcps1.2.5.1.1"><p id="p303mcpsimp"><a name="p303mcpsimp"></a><a name="p303mcpsimp"></a>Item</p>
</th>
<th class="cellrowborder" valign="top" width="15.98%" id="mcps1.2.5.1.2"><p id="p305mcpsimp"><a name="p305mcpsimp"></a><a name="p305mcpsimp"></a>Version</p>
</th>
<th class="cellrowborder" valign="top" width="35.02%" id="mcps1.2.5.1.3"><p id="p307mcpsimp"><a name="p307mcpsimp"></a><a name="p307mcpsimp"></a>Description</p>
</th>
<th class="cellrowborder" valign="top" width="34%" id="mcps1.2.5.1.4"><p id="p309mcpsimp"><a name="p309mcpsimp"></a><a name="p309mcpsimp"></a>How to Obtain</p>
</th>
</tr>
</thead>
<tbody><tr id="row321mcpsimp"><td class="cellrowborder" valign="top" width="15%" headers="mcps1.2.5.1.1 "><p id="p323mcpsimp"><a name="p323mcpsimp"></a><a name="p323mcpsimp"></a>Python</p>
</td>
<td class="cellrowborder" valign="top" width="15.98%" headers="mcps1.2.5.1.2 "><p id="p325mcpsimp"><a name="p325mcpsimp"></a><a name="p325mcpsimp"></a>3.11.x</p>
</td>
<td class="cellrowborder" valign="top" width="35.02%" headers="mcps1.2.5.1.3 "><p id="p327mcpsimp"><a name="p327mcpsimp"></a><a name="p327mcpsimp"></a>Python is an auxiliary tool used during the build of TF Serving. It is used to automatically download dependencies and configure the environment.</p>
</td>
<td class="cellrowborder" valign="top" width="34%" headers="mcps1.2.5.1.4 "><p id="p329mcpsimp"><a name="p329mcpsimp"></a><a name="p329mcpsimp"></a>Install it using a Yum repository.</p>
</td>
</tr>
<tr id="row330mcpsimp"><td class="cellrowborder" valign="top" width="15%" headers="mcps1.2.5.1.1 "><p id="p332mcpsimp"><a name="p332mcpsimp"></a><a name="p332mcpsimp"></a>CMake</p>
</td>
<td class="cellrowborder" valign="top" width="15.98%" headers="mcps1.2.5.1.2 "><p id="p1890631664615"><a name="p1890631664615"></a><a name="p1890631664615"></a>3.27.9</p>
</td>
<td class="cellrowborder" valign="top" width="35.02%" headers="mcps1.2.5.1.3 "><p id="p336mcpsimp"><a name="p336mcpsimp"></a><a name="p336mcpsimp"></a>CMake is the build tool of TF Serving. The CMake version must be 3.22.0 or later.</p>
</td>
<td class="cellrowborder" valign="top" width="34%" headers="mcps1.2.5.1.4 "><p id="p338mcpsimp"><a name="p338mcpsimp"></a><a name="p338mcpsimp"></a>Install it using a Yum repository.</p>
</td>
</tr>
<tr id="row339mcpsimp"><td class="cellrowborder" valign="top" width="15%" headers="mcps1.2.5.1.1 "><p id="p341mcpsimp"><a name="p341mcpsimp"></a><a name="p341mcpsimp"></a>GCC/G++</p>
</td>
<td class="cellrowborder" valign="top" width="15.98%" headers="mcps1.2.5.1.2 "><p id="p343mcpsimp"><a name="p343mcpsimp"></a><a name="p343mcpsimp"></a>12.3.1</p>
</td>
<td class="cellrowborder" valign="top" width="35.02%" headers="mcps1.2.5.1.3 "><p id="p345mcpsimp"><a name="p345mcpsimp"></a><a name="p345mcpsimp"></a>GNU Compiler Collection (GCC) serves as the compiler for compiling TF Serving into an executable file.</p>
</td>
<td class="cellrowborder" valign="top" width="34%" headers="mcps1.2.5.1.4 "><p id="p347mcpsimp"><a name="p347mcpsimp"></a><a name="p347mcpsimp"></a>Install it using a Yum repository.</p>
</td>
</tr>
<tr id="row348mcpsimp"><td class="cellrowborder" valign="top" width="15%" headers="mcps1.2.5.1.1 "><p id="p8254102818211"><a name="p8254102818211"></a><a name="p8254102818211"></a>Bazel</p>
</td>
<td class="cellrowborder" valign="top" width="15.98%" headers="mcps1.2.5.1.2 "><p id="p525314288218"><a name="p525314288218"></a><a name="p525314288218"></a>6.5.0</p>
</td>
<td class="cellrowborder" valign="top" width="35.02%" headers="mcps1.2.5.1.3 "><p id="p10253172810219"><a name="p10253172810219"></a><a name="p10253172810219"></a>Bazel is a powerful build system that enables fast and scalable building. The Bazel version must be 6.4.0 or later.</p>
</td>
<td class="cellrowborder" valign="top" width="34%" headers="mcps1.2.5.1.4 "><p id="p823317281218"><a name="p823317281218"></a><a name="p823317281218"></a><a href="https://releases.bazel.build/6.5.0/release/bazel-6.5.0-dist.zip" target="_blank" rel="noopener noreferrer">Link</a></p>
</td>
</tr>
</tbody>
</table>

### Compilation Environment Setup

**Obtaining the KDNN Software Package**

1. Decompress the obtained KDNN software package to obtain the binary RPM package.
2. Install the RPM package of KDNN.

   ```shell
   rpm -ivh boostcore-kdnn-xxxx.aarch64.rpm
   ```

   After the installation is complete, the header files, static library files, and dynamic library files are stored in `/usr/local/kdnn/include`, `/usr/local/kdnn/lib/threadpool`, and `/usr/local/kdnn/lib/omp directories`, respectively.

   In the preceding command, *xxxx* indicates the version number.

### Compilation and Installation

**Adapting TensorFlow for KDNN**

Adapt TensorFlow by following the instructions in [Adapting TensorFlow for KDNN](https://www.hikunpeng.com/document/detail/en/kunpengaccel/kail/kaiOperl/docs/en/kdnn/best_practices.md#adapting-tensorflow-for-KDNN) in the _KDNN Best Practices_. Replace the patch with `0001-tensorflow_2.15.0-optimize.patch`, and in Step 4, download the package from `boostkit/tensorflow.git` using the branch tag `v1.0.0`.

## TensorFlow ANNC Static Graph Fusion

### Verified Environment

**Hardware Requirements<a name="section230mcpsimp"></a>**

[**Table 1**](#_table38928044) lists the verified hardware environment.

**Table 1** Hardware requirement

<a name="_table38928044"></a>

<table><thead align="left"><tr id="row239mcpsimp"><th class="cellrowborder" valign="top" width="25%" id="mcps1.2.3.1.1"><p id="p241mcpsimp"><a name="p241mcpsimp"></a><a name="p241mcpsimp"></a>Item</p>
</th>
<th class="cellrowborder" valign="top" width="75%" id="mcps1.2.3.1.2"><p id="p243mcpsimp"><a name="p243mcpsimp"></a><a name="p243mcpsimp"></a>Description</p>
</th>
</tr>
</thead>
<tbody><tr id="row245mcpsimp"><td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.3.1.1 "><p id="p247mcpsimp"><a name="p247mcpsimp"></a><a name="p247mcpsimp"></a>CPU</p>
</td>
<td class="cellrowborder" valign="top" width="75%" headers="mcps1.2.3.1.2 "><p id="p249mcpsimp"><a name="p249mcpsimp"></a><a name="p249mcpsimp"></a>Kunpeng 950 processor</p>
</td>
</tr>
</tbody>
</table>

**OS Requirements<a name="section250mcpsimp"></a>**

[Table 2](#_d0e164) lists the verified OSs.

**Table 2** OSs

<a name="_d0e164"></a>

<table><thead align="left"><tr id="row261mcpsimp"><th class="cellrowborder" valign="top" width="15%" id="mcps1.2.5.1.1"><p id="p263mcpsimp"><a name="p263mcpsimp"></a><a name="p263mcpsimp"></a>Item</p>
</th>
<th class="cellrowborder" valign="top" width="16%" id="mcps1.2.5.1.2"><p id="p265mcpsimp"><a name="p265mcpsimp"></a><a name="p265mcpsimp"></a>Version</p>
</th>
<th class="cellrowborder" valign="top" width="35%" id="mcps1.2.5.1.3"><p id="p267mcpsimp"><a name="p267mcpsimp"></a><a name="p267mcpsimp"></a>Description</p>
</th>
<th class="cellrowborder" valign="top" width="34%" id="mcps1.2.5.1.4"><p id="p269mcpsimp"><a name="p269mcpsimp"></a><a name="p269mcpsimp"></a>How to Obtain</p>
</th>
</tr>
</thead>
<tbody><tr id="row271mcpsimp"><td class="cellrowborder" valign="top" width="15%" headers="mcps1.2.5.1.1 "><p id="p273mcpsimp"><a name="p273mcpsimp"></a><a name="p273mcpsimp"></a>OS</p>
</td>
<td class="cellrowborder" valign="top" width="16%" headers="mcps1.2.5.1.2 "><p id="p275mcpsimp"><a name="p275mcpsimp"></a><a name="p275mcpsimp"></a>openEuler 24.03 LTS SP3</p>
</td>
<td class="cellrowborder" valign="top" width="35%" headers="mcps1.2.5.1.3 "><p id="p277mcpsimp"><a name="p277mcpsimp"></a><a name="p277mcpsimp"></a>When installing an OS, choose <code>Minimal Install</code> and select <code>Development Tools</code> to minimize manual operations.</p>
</td>
<td class="cellrowborder" valign="top" width="34%" headers="mcps1.2.5.1.4 "><p id="p279mcpsimp"><a name="p279mcpsimp"></a><a name="p279mcpsimp"></a><a href="https://repo.openeuler.org/openEuler-24.03-LTS-SP3/ISO/aarch64/" target="_blank" rel="noopener noreferrer">Link</a></p>
</td>
</tr>
<tr id="row281mcpsimp"><td class="cellrowborder" valign="top" width="15%" headers="mcps1.2.5.1.1 "><p id="p283mcpsimp"><a name="p283mcpsimp"></a><a name="p283mcpsimp"></a>Kernel</p>
</td>
<td class="cellrowborder" valign="top" width="16%" headers="mcps1.2.5.1.2 "><p id="p285mcpsimp"><a name="p285mcpsimp"></a><a name="p285mcpsimp"></a>6.6.0</p>
</td>
<td class="cellrowborder" valign="top" width="35%" headers="mcps1.2.5.1.3 "><p id="p287mcpsimp"><a name="p287mcpsimp"></a><a name="p287mcpsimp"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="34%" headers="mcps1.2.5.1.4 "><p id="p289mcpsimp"><a name="p289mcpsimp"></a><a name="p289mcpsimp"></a>Included in the OS image</p>
</td>
</tr>
</tbody>
</table>

**Software Requirements<a name="section290mcpsimp"></a>**

[Table 3](#_table237115053311) lists the verified software dependencies.

**Table 3** Software requirements

<a name="_table237115053311"></a>

<table><thead align="left"><tr id="row301mcpsimp"><th class="cellrowborder" valign="top" width="15%" id="mcps1.2.5.1.1"><p id="p303mcpsimp"><a name="p303mcpsimp"></a><a name="p303mcpsimp"></a>Item</p>
</th>
<th class="cellrowborder" valign="top" width="15.98%" id="mcps1.2.5.1.2"><p id="p305mcpsimp"><a name="p305mcpsimp"></a><a name="p305mcpsimp"></a>Version</p>
</th>
<th class="cellrowborder" valign="top" width="35.02%" id="mcps1.2.5.1.3"><p id="p307mcpsimp"><a name="p307mcpsimp"></a><a name="p307mcpsimp"></a>Description</p>
</th>
<th class="cellrowborder" valign="top" width="34%" id="mcps1.2.5.1.4"><p id="p309mcpsimp"><a name="p309mcpsimp"></a><a name="p309mcpsimp"></a>How to Obtain</p>
</th>
</tr>
</thead>
<tbody><tr id="row321mcpsimp"><td class="cellrowborder" valign="top" width="15%" headers="mcps1.2.5.1.1 "><p id="p323mcpsimp"><a name="p323mcpsimp"></a><a name="p323mcpsimp"></a>Python</p>
</td>
<td class="cellrowborder" valign="top" width="15.98%" headers="mcps1.2.5.1.2 "><p id="p325mcpsimp"><a name="p325mcpsimp"></a><a name="p325mcpsimp"></a>3.11.x</p>
</td>
<td class="cellrowborder" valign="top" width="35.02%" headers="mcps1.2.5.1.3 "><p id="p327mcpsimp"><a name="p327mcpsimp"></a><a name="p327mcpsimp"></a>Python is an auxiliary tool used during the build of TF Serving. It is used to automatically download dependencies and configure the environment.</p>
</td>
<td class="cellrowborder" valign="top" width="34%" headers="mcps1.2.5.1.4 "><p id="p329mcpsimp"><a name="p329mcpsimp"></a><a name="p329mcpsimp"></a>Install it using a Yum repository.</p>
</td>
</tr>
<tr id="row330mcpsimp"><td class="cellrowborder" valign="top" width="15%" headers="mcps1.2.5.1.1 "><p id="p332mcpsimp"><a name="p332mcpsimp"></a><a name="p332mcpsimp"></a>CMake</p>
</td>
<td class="cellrowborder" valign="top" width="15.98%" headers="mcps1.2.5.1.2 "><p id="p1890631664615"><a name="p1890631664615"></a><a name="p1890631664615"></a>3.27.9</p>
</td>
<td class="cellrowborder" valign="top" width="35.02%" headers="mcps1.2.5.1.3 "><p id="p336mcpsimp"><a name="p336mcpsimp"></a><a name="p336mcpsimp"></a>CMake is the build tool of TF Serving. The CMake version must be 3.22.0 or later.</p>
</td>
<td class="cellrowborder" valign="top" width="34%" headers="mcps1.2.5.1.4 "><p id="p338mcpsimp"><a name="p338mcpsimp"></a><a name="p338mcpsimp"></a>Install it using a Yum repository.</p>
</td>
</tr>
<tr id="row339mcpsimp"><td class="cellrowborder" valign="top" width="15%" headers="mcps1.2.5.1.1 "><p id="p341mcpsimp"><a name="p341mcpsimp"></a><a name="p341mcpsimp"></a>GCC/G++</p>
</td>
<td class="cellrowborder" valign="top" width="15.98%" headers="mcps1.2.5.1.2 "><p id="p343mcpsimp"><a name="p343mcpsimp"></a><a name="p343mcpsimp"></a>12.3.1</p>
</td>
<td class="cellrowborder" valign="top" width="35.02%" headers="mcps1.2.5.1.3 "><p id="p345mcpsimp"><a name="p345mcpsimp"></a><a name="p345mcpsimp"></a>GNU Compiler Collection (GCC) serves as the compiler for compiling TF Serving into an executable file.</p>
</td>
<td class="cellrowborder" valign="top" width="34%" headers="mcps1.2.5.1.4 "><p id="p347mcpsimp"><a name="p347mcpsimp"></a><a name="p347mcpsimp"></a>Install it using a Yum repository.</p>
</td>
</tr>
<tr id="row348mcpsimp"><td class="cellrowborder" valign="top" width="15%" headers="mcps1.2.5.1.1 "><p id="p8254102818211"><a name="p8254102818211"></a><a name="p8254102818211"></a>Bazel</p>
</td>
<td class="cellrowborder" valign="top" width="15.98%" headers="mcps1.2.5.1.2 "><p id="p525314288218"><a name="p525314288218"></a><a name="p525314288218"></a>6.5.0</p>
</td>
<td class="cellrowborder" valign="top" width="35.02%" headers="mcps1.2.5.1.3 "><p id="p10253172810219"><a name="p10253172810219"></a><a name="p10253172810219"></a>Bazel is a powerful build system that enables fast and scalable building. The Bazel version must be 6.4.0 or later.</p>
</td>
<td class="cellrowborder" valign="top" width="34%" headers="mcps1.2.5.1.4 "><p id="p823317281218"><a name="p823317281218"></a><a name="p823317281218"></a><a href="https://releases.bazel.build/6.5.0/release/bazel-6.5.0-dist.zip" target="_blank" rel="noopener noreferrer">Link</a></p>
</td>
</tr>
</tbody>
</table>

### Performing Compilation and Installation

**Applying ANNC Static Graph Fusion Patches**

1. Install basic software.

   ```bash
   yum install gcc g++ zip python vim tar wget unzip 
   ```

2. Install Bazel.

   Install Bazel by following the instructions in [Installing Bazel](https://www.hikunpeng.com/document/detail/en/SRA/ecosystemEnable/TensorFlow/kunpengtensorflow_02_0008.html) in the _TensorFlow Porting Guide_. Bazel is required for TensorFlow compilation. For TensorFlow 2.15.0, download Bazel 6.5.0.
3. Download the TensorFlow source code.

   Download open-source TensorFlow 2.15.0 from GitHub.

   ```bash
   git clone -b v2.15.0 https://github.com/tensorflow/tensorflow.git 
   ```

4. Download the optimization patches.

   Download the optimization patch from GitCode and integrate it into the open-source TensorFlow directory.

   ```bash
   git clone -b v1.1.0 https://gitcode.com/boostkit/tensorflow.git sra-tensorflow 
   cp /path/to/sra-tensorflow/0001-tensorflow\_2.15.0-optimize.patch /path/to/tensorflow/
   cp /path/to/sra-tensorflow/0002-tensorflow\_2.15.0-annc-optimize.patch /path/to/tensorflow/
   cd /path/to/tensorflow && patch -p1 < 0001-tensorflow\_2.15.0-optimize.patch && patch -p1 < 0002-tensorflow\_2.15.0-annc-optimize.patch
   ```

5. Install the dependencies.

   ```bash
   yum install patchelf perl python3-devel
   pip3 install numpy==1.24.3
   pip3 install certifi==2023.7.22
   pip3 install requests==2.31.0
   pip3 install grpcio==1.59.0
   pip3 install packaging
   pip3 install wheel
   export C_INCLUDE_PATH=/usr/include/python3.11:$C_INCLUDE_PATH
   export CPLUS_INCLUDE_PATH=/usr/include/python3.11:$CPLUS_INCLUDE_PATH
   ```

   In the preceding commands, <code>/usr/include/python3.11</code> is the directory where <code>Python.h</code> is stored. Replace it with the actual directory in the compilation environment.
6. Build Configurations

   For details about how to configure the compilation options, see [Installation from Source Code](https://www.hikunpeng.com/document/detail/en/SRA/ecosystemEnable/TensorFlow/kunpengtensorflow_02_0009.html) in the _TensorFlow Porting Guide_.

   ```bash
   ./configure
   ```

7. Start the build.

   ```bash
   export TF_PYTHON_VERSION=3.11
   bazel build  //tensorflow/tools/pip_package:build_pip_package
   ```

8. Install the pip package.

   ```bash
   ./bazel-bin/tensorflow/tools/pip_package/build_pip_package ./output
   pip3 install ./output/tensorflow-2.15.0-cp311-cp311-linux_aarch64.whl
   ```

### Verification After Adaptation

1. Go to the `/path/to/tensorflow/tensorflow/python/grappler/embedding_fused_test` directory.
2. Query the names of supported modules.

   ```bash
   python main.py --list
   ```

3. Runs test cases.

   ```bash
   python main.py --op {op_name} --performance_test False
   ```

   If the execution is successful, the installation is successful. `{op_name}` is the query result in the previous step.

## FAQs

If any problem occurs during the compilation and building of TensorFlow and TensorFlow Serving, rectify the fault by following instructions in this section.

- If the message "unable to find valid certification path to requested target" is displayed, you can fix it by following the instructions in [Failed to Verify the Certificate When Compiling TensorFlow 2.13.0 Source Code](https://www.hikunpeng.com/document/detail/en/SRA/ecosystemEnable/TensorFlow/kunpengtensorflow_02_0012.html) in the _TensorFlow Porting Guide_.
- If the message "Error in download_and_extract" is displayed, you can fix it by following the instructions in [Failed to Download the TF-Serving Source Code Dependency](https://www.hikunpeng.com/document/detail/en/SRA/ecosystemEnable/TensorFlowServing/kunpengtfserving_06_0014.html) in the _TensorFlow Serving Porting Guide_. The directory specified by `--distdir` is `/path/to/tensorflow-serving/download`.
- If the system displays a message indicating that the `org_boost` dependency library fails to be pulled, you can fix it by following the instructions in [Failed to Obtain the Dependency of the org_boost Sub-repository](https://www.hikunpeng.com/document/detail/en/SRA/ecosystemEnable/TensorFlowServing/kunpengtfserving_06_0015.html) in the _TensorFlow Serving Porting Guide_.
- If the system displays a message indicating that the Golang website certificate is unavailable, you can fix it by following the instructions in [No Golang Website Certificate](https://www.hikunpeng.com/document/detail/en/SRA/ecosystemEnable/TensorFlowServing/kunpengtfserving_06_0016.html) in the _TensorFlow Serving Porting Guide_.
- If the system displays a message indicating that the Golang download fails during ANNC compilation, you can fix it by following the instructions in [Failed to Download the TF-Serving Source Code Dependency](https://www.hikunpeng.com/document/detail/en/SRA/ecosystemEnable/TensorFlowServing/kunpengtfserving_06_0014.html) in the _TensorFlow Serving Porting Guide_. The directory specified by `--distdir` can be specified in the `build.sh` script.
- If the system displays a message indicating that the `upb.c` file fails to be compiled, you can fix it by following the instructions in [Syntax Error Reported During upb.c Compilation](https://www.hikunpeng.com/document/detail/en/SRA/ecosystemEnable/TensorFlowServing/kunpengtfserving_06_0017.html) in the _TensorFlow Serving Porting Guide_.

## Description

| Release Date | Change History |
| ---------- | -------------- |
| 2026-06-30 | This is the second official release. <ul><li>Added the description for constant folding optimization to the TensorFlow ANNC for graph compilation documentation. </li><li>Added the environment support and installation guide for TensorFlow ANNC static graph fusion. </li></ul> |
| 2026-03-30 | This is the first official release. <ul><li>Added installation steps for TensorFlow with KDNN integration. </li><li>Added the environment support and installation guide for KDNN thread passthrough.</li></ul> |
