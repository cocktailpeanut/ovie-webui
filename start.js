module.exports = {
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        path: ".",
        message: [
          "uv run --project app python ovie_webui.py --port {{port}}"
        ],
        on: [{
          event: "/(http:\\/\\/\\S+)/",
          done: true
        }]
      }
    },
    {
      method: "local.set",
      params: {
        url: "{{input.event[1]}}"
      }
    }
  ]
}
