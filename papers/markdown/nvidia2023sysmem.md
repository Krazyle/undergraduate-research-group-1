# System Memory Fallback for Stable Diffusion

Updated 10/31/2023 06:05 AM

System Memory Fallback for Stable Diffusion

------------------------------------------------------------------------

In driver 536.40, we implemented a new method to allow an application to use shared memory in cases that exhausted the GPU memory. This enabled applications which previously crashed when running out of GPU memory to continue to run, albeit at lower speeds. The switch to use shared memory occurs when running close to maxing out GPU memory to allow for a seamless transition. 

 

Stable Diffusion happens to require close to 6 GB of GPU memory often. This can cause the above mechanism to be invoked for people on 6 GB GPUs, reducing the application speed. In driver 546.01 and above we added a setting to disable the shared memory fallback, which should make performance stable at the risk of a crash if the user uses a setting that requires more GPU memory. This guide will explain how to do this. 

 

Instructions to disable shared memory fallback for Stable Diffusion

1\. Run Stable Diffusion and open task manager. Copy down the path of python.exe that is running. In the screenshot below, the path would be:\
"c:\users\USERNAME\downloads\sd.webui\system\python\python.exe"\
\
![](https://nvidia.custhelp.com/rnt/rnw/img/enduser/aid_5490_01.png)

 

2. Right-click the Windows desktop and select NVIDIA Control Panel as shown in the image below. This will launch the NVIDIA Control Panel.\
\
![](https://nvidia.custhelp.com/rnt/rnw/img/enduser/aid_5490_02.png)

 

3. In the 3D Settings section, click on Manage 3D settings.\
\
![](https://nvidia.custhelp.com/rnt/rnw/img/enduser/aid_5490_03.png)

 

4. Navigate to the Program Settings tab. Click the Add button under programs to customize.\
\
![](https://nvidia.custhelp.com/rnt/rnw/img/enduser/aid_5490_04.png)

 

5. This launches Select a Program.\
\
![](https://nvidia.custhelp.com/rnt/rnw/img/enduser/aid_5490_05.png)\
 

6. Click Browse and locate the Python executable file noted in Step 1. Select it and click Open to confirm the selection.\
\
![](https://nvidia.custhelp.com/rnt/rnw/img/enduser/aid_5490_06.png)

 

7. Click on CUDA - Sysmem Fallback Policy. Set the value to Prefer No Sysmem Fallback.\
\
![](https://nvidia.custhelp.com/rnt/rnw/img/enduser/aid_5490_07.png)

 

8. Click Apply to confirm.

 

9. Re-launch Stable Diffusion. Note that Stable Diffusion must be restarted following these steps for this setting change to be applied.

 

10. You can re-enable system memory fallback at any time by coming back to this tab. Full steps:

a\. Open NVIDIA Control Panel

b\. Under 3D Settings, click Manage 3D Settings 

c\. Navigate to Program Settings tab

d\. Select Stable Diffusion python executable from dropdown

e\. Click on CUDA - Sysmem Fallback Policy and select Driver Default.

f\. Click Apply to confirm.

g. Restart Stable Diffusion if it’s already open.\
 
