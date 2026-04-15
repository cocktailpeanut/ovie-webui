module.exports = {
  run: [{
    method: "shell.run",
    params: {
      message: "git pull"
    }
  }, {
    when: "{{exists('app')}}",
    method: "shell.run",
    params: {
      path: "app",
      message: "git pull"
    }
  }, {
    when: "{{exists('app')}}",
    method: "shell.run",
    params: {
      bluefairy: "off",
      path: "app",
      message: "uv sync --frozen"
    }
  }]
}
