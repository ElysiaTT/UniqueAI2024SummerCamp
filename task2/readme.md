# 第二周实验感想
## 大模型的部署

使用平台：
modelscope（失败）、autodl（成功）

参考文献：

1.https://zhuanlan.zhihu.com/p/687577203?utm_medium=social&utm_psn=1802709362755629056&utm_source=wechat_session（成功，主要实现的就是这个Qwen1.5-7B-Chat-GPTQ-Int4 部署）

2.https://blog.csdn.net/weixin_42118737/article/details/138858898（最开始尝试，但是因为文章中是本地化部署，而刚开始在modelscope上免费实例运行没有miniconda导致没有虚拟环境）

3.https://blog.csdn.net/chrnhao/article/details/136284249?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522172209062216800180660540%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=172209062216800180660540&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-2-136284249-null-null.142^v100^pc_search_result_base3&utm_term=modelscope%E4%B8%8B%E8%BD%BDqwen1.5-1.8b&spm=1018.2226.3001.4187
（在modelscope上，这个模型当时下载的感觉不是很完整，而且很大，有可能是没下载到数据盘导致内存爆炸，后面就放弃了）

4.https://blog.csdn.net/qq_36366417/article/details/138648791?ops_request_misc=&request_id=&biz_id=102&utm_term=modelscope%E4%B8%8B%E8%BD%BDqwen1.5-1.8b&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-8-138648791.142^v100^pc_search_result_base3&spm=1018.2226.3001.4187
（教程不详细，配了一次就放弃了）

5.https://blog.csdn.net/huiguo_/article/details/132584559?ops_request_misc=&request_id=&biz_id=102&utm_term=%E9%80%9A%E4%B9%89%E5%8D%83%E9%97%AE%E4%BA%91%E5%B9%B3%E5%8F%B0%E9%83%A8%E7%BD%B2&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-132584559.142^v100^pc_search_result_base3&spm=1018.2226.3001.4187
（这个就差临门一脚，当时它里面的部署1的demo只能直接输出示例，于是我直接去找了部署2，但是最后输出的url打不开，我在此纠结了很久，最后还是换到了第1个方法，实现了Qwen1.5-7B-Chat-GPTQ-Int4 部署。回看这个方法，我知道那个时候我只要再做一步就可以在web上实现部署）
