# AlphaAgent 前端演示

本目录是 AlphaAgent 的本地 React/TypeScript 单页演示。页面通过 `http://127.0.0.1:8000/api/demo` 读取正式回测指标与逐日选股建议。

推荐直接双击项目根目录的 `启动AlphaAgent演示.bat`，它会同时启动策略服务与前端并打开浏览器。也可以手动启动：

```powershell
python start_demo.py
```

前端单独开发：

```powershell
cd frontend
npm run dev
```

生产构建与测试：

```powershell
npm test
```

如需修改后端地址，设置 `NEXT_PUBLIC_API_BASE`，默认值为 `http://127.0.0.1:8000/api/demo`。
