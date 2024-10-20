## > 注 #########################################################
##                No guarantee! Use with caution!             ##
##                     © 1479065488@qq.com                    ##
## 注 < #########################################################

# 0 自动设置工作目录到当前文件所在目录
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# 1 提前安装好以下包 ----
suppressMessages({
  library(shiny)
  library(shinydashboard)
  library(readr)
  library(dplyr)
  library(ggplot2)
  library(svglite)
  library(ggraph)
  library(tidygraph)
  library(igraph)
})

# 2 定义 UI ----
ui <- dashboardPage(
  # 主题
  skin = "green",
  # 名称
  dashboardHeader(
    title = "🕹",
    titleWidth = 300
  ),
  # 侧边栏
  dashboardSidebar(
    width = 300,
    tabItem(
      tabName = "DAG",
      br(),
      box(
        width = 12,
        style = "color: black;",
        fileInput(
          "semTable",
          "☝️ 上传 from，to，weight，p 表格",
          placeholder = ".csv",
          width = "100%"
        )
      ),
      hr(),
      box(
        width = 12,
        style = "color: black;",
        selectInput(
          "selectedLayout",
          "🕸 布局",
          choices = c("sugiyama", "circle", "tree", "grid"),
          selected = "sugiyama"
        ),
        sliderInput(
          "pThreshold",
          "🚫 路径显著性阈值",
          min = 0.05,
          max = 1,
          value = 1,
          step = 0.05,
          ticks = FALSE
        ),
        sliderInput(
          "fontSize",
          "😁 文字大小",
          min = 3,
          max = 11,
          value = 3,
          step = 0.5,
          ticks = FALSE
        ),
        sliderInput(
          "lineWidth",
          "😊 线条粗细",
          min = 1,
          max = 7,
          value = 3,
          step = 0.1,
          ticks = FALSE
        ),
        sliderInput(
          "setCurvature",
          "😌 线条曲率",
          min = -1,
          max = 1,
          value = 0.5,
          step = 0.1,
          ticks = FALSE
        ),
        br(),
        fluidRow(
          column(
            width = 10,
            align = "center",
            downloadLink("downloadSEMSVG", "👉 Graph.svg", class = "download-link")
          )
        ),
        hr(),
        uiOutput("selecteResponseUI"), # 添加的 UI
        selectInput(
          "themeEfPlot",
          "🎨 绘图主题",
          choices = c("theme_minimal", "theme_classic", "theme_bw"),
          selected = "theme_minimal"
        ),
        br(),
        fluidRow(
          column(
            width = 10,
            align = "center",
            downloadLink("downloadEFSVG", "👉 Effects.svg", class = "download-link")
          )
        ),
        hr(),
        sliderInput(
          "saveWd",
          "🤗 保存图宽",
          min = 4,
          max = 15,
          value = 8,
          step = 0.5,
          ticks = FALSE
        ),
        sliderInput(
          "saveHt",
          "🤔 保存图高",
          min = 4,
          max = 15,
          value = 5,
          step = 0.5,
          ticks = FALSE
        )
      ),
      hr(),
      box(
        width = 12,
        style = "color: black;",
        textAreaInput(
          "inLayoutTx",
          "",
          value = "此处：1、粘贴曾保存过的布局，以快速恢复到某次调整所保存的状态；2、格式是 name x y；3、存图时，会同时保存一个布局 .csv 文件。",
          width = "100%",
          height = "200px"
        )
      )
    )
  ),
  # 主体
  dashboardBody(
    tags$head(
      tags$style(
        HTML("
          .download-link {
            color: green !important;
            display: flex;
            justify-content: center;
            align-items: center;
          }
          .content-wrapper, .right-side {
            height: auto !important;
            overflow: auto !important;
          }
        ")
      )
    ),
    tabsetPanel(
      id = "tabs",
      tabPanel(
        title = "", # SEM 图
        plotOutput("graphSEM", click = "plotClick", height = "600px"),
        hr(),
        plotOutput("plotEffects", height = "600px"),
        icon = icon("circle-nodes") # circle-nodes, share-nodes, square-poll-vertical
      ),
      tabPanel(
        title = "", # 简介
        htmlOutput("appNotes", height = "600px"),
        icon = icon("circle-question") # circle-exclamation, circle-question
      )
    )
  )
)

# 3 定义 SERVER ----
server <- function(input, output) {
  # 重新映射
  reMap <- function(x, mn, mx) {
    (x - min(x)) * (mx - mn) / (max(x) - min(x)) + mn
  }

  #  P 值转为星号
  p2Star <- function(p) {
    symnum(
      p,
      corr = F,
      na = F,
      cutpoints = c(0, 0.001, 0.01, 0.05, 0.1, 1),
      symbols = c("***", "**", "*", ".", " ")
    )
  }

  # 响应式表达式（会随操作而改变变）
  reactVal <- reactiveValues()

  # 基础 SEM 图
  rawGraph <- reactive({
    req(input$semTable)
    cfl <- input$semTable
    read_csv(
      cfl$datapath,
      show_col_types = FALSE
    ) |>
      as_tbl_graph(directed = T)
  })

  # 重新布局
  reLayoutDf <- reactive({
    inLayoutDf <- create_layout(
      rawGraph(),
      layout = input$selectedLayout
    ) |>
      mutate(
        x = round(reMap(x, -9, 9)),
        y = round(reMap(y, -9, 9))
      )

    if (!input$inLayoutTx == "") { # 检查输入是否为空
      req(input$inLayoutTx) # 确保输入值存在
      # 尝试读取文本文件并处理数据
      tryCatch(
        {
          layoutByTx <- readr::read_delim(
            input$inLayoutTx,
            show_col_types = F,
            progress = F
          )
          layoutByTxRow <- readr::problems(layoutByTx)[["row"]] # layoutByTxRow <- layoutByTx |> problems() |> pull(row)
          if (is.null(layoutByTxRow)) {
            layoutNoBadRow <- slice(layoutByTx, -layoutByTxRow)
          } else {
            layoutNoBadRow <- layoutByTx
          }
          inLayoutDf <- inLayoutDf |>
            left_join(layoutNoBadRow, by = "name") |>
            mutate(
              x = coalesce(x.y, x.x),
              y = coalesce(y.y, y.x)
            ) |>
            dplyr::select(name, x, y)
        },
        error = function(e) NULL
      )
    }
    inLayoutDf
  })

  # 响应式表达式（会随操作而改变变）
  observe({
    reactVal$reLayoutDf <- reLayoutDf()
    reactVal$selectedNode <- NULL
    reactVal$clickOdEvn <- 0
    reactVal$layoutClick <- NULL
  })

  # 根据输入（点击或手动指定）改变的布局
  basicLayout <- reactive({
    create_layout(
      rawGraph(),
      layout = "manual",
      x = reactVal$reLayoutDf$x,
      y = reactVal$reLayoutDf$y
    )
  })

  # 字体
  setFont <- "Prompt" # "sans"

  # 基础 SEM 图
  basicGraph <- reactive({
    nodeNames <- igraph::V(rawGraph())$name
    edgeList <- igraph::as_edgelist(rawGraph())
    basicLayoutY <- basicLayout()$y
    names(basicLayoutY) <- nodeNames
    needArc <- basicLayoutY[edgeList[, 1]] == basicLayoutY[edgeList[, 2]]
    basicLayout() |>
      ggraph() +
      geom_edge_diagonal(
        strength = 0,
        # strength = input$setCurvature,
        aes(
          filter = ifelse(is.na(p), 1, p) < input$pThreshold & !needArc, # --- 设置 P value 的阈值
          edge_width = abs(weight),
          edge_color = case_when(
            p < 0.05 & weight > 0 ~ "+",
            p < 0.05 & weight < 0 ~ "-",
            T ~ "P > 0.05"
          ),
          label = if (class(weight) == "numeric") {
            paste0(round(weight, 2), gsub("[.]", "", p2Star(p)))
          } else {
            paste(weight)
          },
          start_cap = label_rect(node1.name, padding = margin(10, 10, 10, 10)),
          end_cap = label_rect(node2.name, padding = margin(10, 10, 10, 10))
        ),
        label_size = input$fontSize,
        family = setFont,
        angle_calc = "along",
        hjust = 1,
        vjust = -0.5,
        arrow = arrow(60, length = unit(7, "pt"), type = "closed"), # --- 设置实心箭头
        # arrow = arrow(60, length = unit(11, "pt"), type = "open"), # --- 设置钩状箭头
        lineend = "butt",
        linejoin = "mitre"
      ) +
      geom_edge_arc(
        strength = input$setCurvature,
        aes(
          filter = ifelse(is.na(p), 1, p) < input$pThreshold & needArc, # --- 设置 P value 的阈值
          edge_width = abs(weight),
          edge_color = case_when(
            p < 0.05 & weight > 0 ~ "+",
            p < 0.05 & weight < 0 ~ "-",
            T ~ "P > 0.05"
          ),
          label = if (class(weight) == "numeric") {
            paste0(round(weight, 2), gsub("[.]", "", p2Star(p)))
          } else {
            paste(weight)
          },
          start_cap = label_rect(node1.name, padding = margin(10, 10, 10, 10)),
          end_cap = label_rect(node2.name, padding = margin(10, 10, 10, 10))
        ),
        label_size = input$fontSize,
        family = setFont,
        angle_calc = "along",
        hjust = 1,
        vjust = -0.5,
        arrow = arrow(60, length = unit(7, "pt"), type = "closed"), # --- 设置实心箭头
        # arrow = arrow(60, length = unit(11, "pt"), type = "open"), # --- 设置钩状箭头
        lineend = "butt",
        linejoin = "mitre"
      ) +
      scale_edge_width_continuous(
        name = NULL,
        range = c(0.3, input$lineWidth)
      ) +
      scale_edge_color_manual(
        name = NULL,
        values = c(
          "+" = "#009E73", # --- 默认使用了色盲友好颜色
          "-" = "#D55E00", # --- 默认使用了色盲友好颜色
          "P > 0.05" = "grey"
        )
      ) + # --- 设置路径颜色
      geom_node_text(
        aes(label = name),
        size = input$fontSize * 1.3,
        family = setFont,
        fontface = "bold",
        show.legend = F
      ) +
      list(
        if (!is.null(reactVal$selectedNode)) {
          geom_point(
            data = filter(reactVal$reLayoutDf, name == reactVal$selectedNode),
            aes(x, y),
            size = 11,
            shape = 21,
            color = "orange1", fill = alpha("orange1", 0.1),
            stroke = 1
          )
        } else {
          NULL
        }
      ) +
      labs(
        x = NULL,
        y = NULL
      ) +
      theme_minimal(
        base_family = setFont,
        base_size = 14
      ) +
      theme(
        panel.grid = element_blank()
      )
  })

  # 输出 SEM 图
  output$graphSEM <- renderPlot({
    xRefPts <- seq(-10, 10, 1)
    yRefPts <- seq(-10, 10, 1)
    xAxBrks <- seq(-10, 10, 5)
    yAxBrks <- seq(-10, 10, 5)
    basicGraph() +
      geom_point(
        data = expand.grid(x1 = xRefPts, y1 = yRefPts),
        aes(x1, y1),
        size = 0.1,
        color = "grey50"
      ) +
      geom_point(
        data = rbind(
          data.frame(x = c(-10, 0, 10), y = c(0, 0, 0)),
          data.frame(x = rep(range(xAxBrks), length(yAxBrks)), y = yAxBrks),
          data.frame(x = xAxBrks, y = rep(range(yAxBrks), length(xAxBrks)))
        ),
        aes(x, y),
        shape = 3,
        size = 2,
        stroke = 0.7,
        color = "grey50"
      ) +
      annotate(
        geom = "text",
        x = 0,
        y = 12,
        label = c("1、点击选取想要移动的节点；2、再点击其他位置，则移动此节点到新位置。"),
        size = 7,
        color = "grey",
        family = setFont
      ) +
      scale_x_continuous(
        breaks = xAxBrks,
        sec.axis = sec_axis(~.x),
        expand = expansion(mult = c(0, 0))
      ) +
      scale_y_continuous(
        breaks = yAxBrks,
        sec.axis = sec_axis(~.x),
        expand = expansion(mult = c(0, 0))
      ) +
      coord_cartesian(
        xlim = c(-11, 11),
        ylim = c(-11, 13),
        clip = "off"
      )
  })

  # 获取拓扑排序的节点名称
  dagTopo <- reactive({
    semDag <- rawGraph()
    igraph::topo_sort(semDag)$name
  })

  # 活动选择控件
  output$selecteResponseUI <- renderUI({
    selectInput(
      "selecteResponse",
      "🎯 响应变量", # Response
      choices = dagTopo(),
      selected = tail(dagTopo())
    )
  })

  # 获取效应
  effectsDf <- reactive({
    # 获取用户选择的节点
    req(input$selecteResponse)
    tempToNode <- input$selecteResponse

    # DAG
    semDag <- rawGraph()
    dagTopoVal <- dagTopo()[seq_along(dagTopo()) < which(dagTopo() == tempToNode)]
    lenTopo <- length(dagTopoVal)

    # 初始化 vector 以存储路径的直接和间接效应
    dEffVector <- NA
    indEffVector <- NA
    fromNode <- NA

    # 循环遍历拓扑排序的节点
    for (j in seq_along(dagTopoVal)) {
      # 所有简单路径
      allSimplePaths <- igraph::all_simple_paths(semDag, from = dagTopoVal[j], to = tempToNode)

      # 初始化间接效应的变量
      directEf <- 0
      indirectEfSum <- 0

      # 遍历所有路径
      # 如果路径长度大于 2，则为间接路径，需要相乘再求和
      # 如果路径长度为 2，则为直接路径，直接取宽度
      for (i in allSimplePaths) {
        eWd <- E(semDag, path = i)$weight
        # 路径长度为 0 则没有连接
        if (length(i) == 0) {
          NULL
          # 路径长度为 2 直接取宽度
        } else if (length(i) == 2) {
          directEf <- eWd
          tempFromNode <- names(i)[1]
        } else {
          # 路径长度大于 2 需要相乘再求和
          indirectEfSum <- indirectEfSum + prod(eWd)
          tempFromNode <- names(i)[1]
        }
      }

      # 存储直接效应和间接效应
      dEffVector[j] <- directEf
      indEffVector[j] <- indirectEfSum
      fromNode[j] <- tempFromNode
    }

    # 存储结果
    data.frame(
      from = fromNode,
      to = tempToNode,
      Direct = dEffVector,
      Indirect = indEffVector
    ) |>
      tidyr::pivot_longer(cols = c(Direct, Indirect)) # |>
    # dplyr::mutate(value = ifelse(value == 0, NA, value))
  })

  # 绘制效应图
  effectsPlot <- reactive({
    effectsDf() |>
      ggplot(aes(from, value, fill = ifelse(value > 0, "+", "-"), lty = name)) +
      geom_col(color = "black") +
      list(
        if (input$themeEfPlot == "theme_minimal") {
          geom_text(
            aes(label = ifelse(value == 0, "", round(value, 3))),
            position = position_stack(vjust = 0.5),
            family = setFont
          )
        } else {
          geom_hline(yintercept = 0, lty = 1, lwd = 0.5)
        }
      ) +
      scale_fill_manual(
        name = NULL,
        values = c(
          "+" = "#009E73", # --- 默认使用了色盲友好颜色
          "-" = "#D55E00" # --- 默认使用了色盲友好颜色
        )
      ) +
      scale_linetype_discrete(
        guide = guide_legend(
          title = NULL,
          override.aes = list(fill = NA)
        )
      ) +
      labs(
        x = "Predictors",
        y = "Effects"
      ) +
      facet_grid(. ~ paste("Response:", to)) +
      get(input$themeEfPlot)(
        base_family = setFont,
        base_size = 14,
        base_rect_size = 1,
        base_line_size = 0.5
      ) +
      theme(
        panel.grid = element_blank(),
        strip.background = element_blank()
      )
  })
  output$plotEffects <- renderPlot({
    effectsPlot()
  })

  # 此处是 ChatGPT 3.5 重新修改过代码，似乎不再依赖 ggraph 的版本，都可以运行了
  observeEvent(input$plotClick, { # 当用户点击图形时触发事件
    reactVal$clickOdEvn <- reactVal$clickOdEvn + 1 # 每次点击事件发生时，增加点击事件的计数器
    if (reactVal$clickOdEvn %% 2 == 1) { # 奇数次点击时
      # 计算所有点与当前点击点之间的欧氏距离，并找到最近的点的索引
      distances <- sqrt((reactVal$reLayoutDf$x - input$plotClick$x)^2 + (reactVal$reLayoutDf$y - input$plotClick$y)^2)
      nearest_index <- which.min(distances)
      # 将最近点的名称存储在 reactVal$selectedNode 中
      reactVal$selectedNode <- reactVal$reLayoutDf$name[nearest_index]
    } else { # 偶数次点击时
      if (!is.null(reactVal$selectedNode) && !is.na(reactVal$selectedNode)) { # 检查是否有选定的点，并且选定的点不是缺失值
        # 更新选中点的坐标为当前点击点的坐标，并四舍五入
        reactVal$reLayoutDf[reactVal$reLayoutDf$name == reactVal$selectedNode, c("x", "y")] <- lapply(input$plotClick[c("x", "y")], function(z) round(as.numeric(z), 0))
        reactVal$selectedNode <- NULL # 更新坐标后，取消选中
      }
    }
    reactVal$layoutClick <- input$plotClick # 更新上次点击的点坐标
  })

  # 要保存的 SEM 图
  plot2Save <- reactive({
    basicGraph() +
      coord_cartesian(
        xlim = c(NA, NA),
        ylim = c(NA, NA),
        clip = "off"
      ) +
      theme(
        axis.text = element_blank()
      )
  })

  # 保存图像
  saveGraph <- function(filename, plotFunction, width, height) {
    ggsave(
      filename,
      plotFunction(),
      width = width,
      height = height
    )
  }

  # Save SEM graph
  output$downloadSEMSVG <- downloadHandler(
    filename = function() {
      "Graph.svg"
    },
    content = function(file) {
      saveGraph(file, plot2Save, input$saveWd, input$saveHt)
      # 保存图形布局
      basicLayout() |>
        dplyr::select(name, x, y) |>
        write.csv(file = "Layout.csv", row.names = FALSE)
    }
  )

  # Save effects graph
  output$downloadEFSVG <- downloadHandler(
    filename = function() {
      "Effects.svg"
    },
    content = function(file) {
      saveGraph(file, effectsPlot, input$saveWd, input$saveHt)
      # 保存效应数据
      effectsDf() |> 
        write.csv(file = "Effects.csv", row.names = FALSE)
    }
  )

  # 简介页面
  output$appNotes <- renderUI({
    HTML(
      "
      <br>
      简言之，“先点选，再点移”。即，点击选取想要移动的节点，再点击其他位置，则移动此节点到新位置（视频教程<a href = '...'>请点此链接</a>）。<br>
      <br>
      No guarantee! Use with caution! © 2024 1479065488@qq.com
      "
    )
  })
}

# 4 运行 ----
shinyApp(ui, server)
