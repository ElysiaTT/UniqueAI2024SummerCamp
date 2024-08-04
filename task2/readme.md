# 第二周实验感想

## MLP

## CNN




## 大模型的部署

### 使用平台：
modelscope（失败）、autodl nvidia 4090（成功）

### 参考文献：

1.https://zhuanlan.zhihu.com/p/687577203?utm_medium=social&utm_psn=1802709362755629056&utm_source=wechat_session（成功，主要实现的就是这个Qwen1.5-7B-Chat-GPTQ-Int4 部署）

2.https://blog.csdn.net/weixin_42118737/article/details/138858898（最开始尝试，但是因为文章中是本地化部署，而刚开始在modelscope上免费实例运行没有miniconda导致没有虚拟环境）

3.https://blog.csdn.net/chrnhao/article/details/136284249?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522172209062216800180660540%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=172209062216800180660540&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-2-136284249-null-null.142^v100^pc_search_result_base3&utm_term=modelscope%E4%B8%8B%E8%BD%BDqwen1.5-1.8b&spm=1018.2226.3001.4187
（在modelscope上，这个模型当时下载的感觉不是很完整，而且很大，有可能是没下载到数据盘导致内存爆炸，后面就放弃了）

4.https://blog.csdn.net/qq_36366417/article/details/138648791?ops_request_misc=&request_id=&biz_id=102&utm_term=modelscope%E4%B8%8B%E8%BD%BDqwen1.5-1.8b&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-8-138648791.142^v100^pc_search_result_base3&spm=1018.2226.3001.4187
（教程不详细，配了一次就放弃了）

5.https://blog.csdn.net/huiguo_/article/details/132584559?ops_request_misc=&request_id=&biz_id=102&utm_term=%E9%80%9A%E4%B9%89%E5%8D%83%E9%97%AE%E4%BA%91%E5%B9%B3%E5%8F%B0%E9%83%A8%E7%BD%B2&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-132584559.142^v100^pc_search_result_base3&spm=1018.2226.3001.4187
（这个就差临门一脚，当时它里面的部署1的demo只能直接输出示例，于是我直接去找了部署2，但是最后输出的url打不开，我在此纠结了很久，最后还是换到了第1个方法，实现了Qwen1.5-7B-Chat-GPTQ-Int4 部署。回看这个方法，我知道那个时候我只要再做一步就可以在web上实现部署，后面再第一个成功案例中我会说明）

### 关键说明

在链接中其实说的非常清楚，但是有几个小点需要注意

1.在文章中说到：从源码安装GPTQ（auto-gptq>=0.5.1），否则极易出现GPTQ无法使用cuda的情况
```
# 从源码安装量化所需GPTQ库
pip install "git+https://github.com/PanQiWei/AutoGPTQ.git@v0.7.1"
```
这里说的gptq大概率只能在云平台上下载一部分，当在web部署是极大可能在交互界面上报错：
```
PackageNotFoundError: No package metadata was found for auto-gptq
```
这时只要在虚拟环境中下载并刷新一下即可：
```
#安装 auto-gptq
pip install auto-gptq
#检查安装
pip show auto-gptq
```
2.可能自己修改了model路径，比如放进了子文件夹或者命名不对。有如下报错：
```
OSError: Incorrect path_or_model_id: '/root/autodl-tmp/qwen/Qwen1.5-7B-Chat-GPTQ-Int4'. Please provide either the path to a local folder or the repo_id of a model on the Hub.
```
回去检查即可：
```
model_path = '/root/autodl-tmp/qwen/Qwen1___5-7B-Chat-GPTQ-Int4'  
print(os.listdir(model_path))  
```
**3.没有把端口映射到本地**
开始时终端会返回一个url如下
![返回值](https://github.com/ElysiaTT/UniqueAI2024SummerCamp/blob/main/task2/return.png)
但是直接点进去没有web界面，会失败：
![失敗する ](https://github.com/ElysiaTT/UniqueAI2024SummerCamp/blob/main/task2/fail.png)

我回看教程时，我是不理解这句话的：“在终端中运行以下命令，启动streamlit服务，并按照 autodl 的指示将端口映射到本地，然后在浏览器中打开链接 http://localhost:6006/ ，即可看到聊天界面。”

直到我把大模型的事情放在一边，学习ai绘画时看到这个文章：https://blog.csdn.net/bossma/article/details/132022385知道这个网站需要在autodl内映射出来才行。

首先要根据文章点击autodl上快捷工具栏内的自定义服务。

但是这个时候的界面与文章中的又不一样，他显示的要打开他所说的界面下载一个autodl ssh隧道工具，打开后是这样的：

![ssh工具](https://github.com/ElysiaTT/UniqueAI2024SummerCamp/blob/main/task2/autodl_ssh.png)

里面的ssh指令与密码在autodl控制界面可以看到：

![ssh工具](https://github.com/ElysiaTT/UniqueAI2024SummerCamp/blob/main/task2/password.png)

然后点击开始代理，就会弹出网站然后直接进去，可能存在1、2中的问题，但是修改完以后就可以了：

![ssh工具](https://github.com/ElysiaTT/UniqueAI2024SummerCamp/blob/main/task2/success.png)

左侧的 max_length 是指模型生成文本的最大长度限制：

![ssh工具](https://github.com/ElysiaTT/UniqueAI2024SummerCamp/blob/main/task2/modify.png)

但是这个好像不太对劲啊:(

## 大模型微调

