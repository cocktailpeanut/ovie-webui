module.exports = {
  run: [
    {
      when: "{{!exists('app')}}",
      method: "shell.run",
      params: {
        message: [
          "git clone --depth 1 https://github.com/kyutai-labs/ovie.git app"
        ]
      }
    },
    {
      method: "shell.run",
      params: {
        bluefairy: "off",
        path: "app",
        message: [
          "uv sync --frozen"
        ]
      }
    }
  ]
}
