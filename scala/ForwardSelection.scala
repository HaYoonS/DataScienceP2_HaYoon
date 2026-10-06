import scala.io.Source
import scalation.mathstat.{MatrixD, VectorD}
import scalation.modeling.Regression

object ForwardSelection:

  def loadCSV(path: String, targetCol: Int): (MatrixD, VectorD, Array[String]) =
    val src = Source.fromFile(path)
    val lines = try src.getLines().toArray finally src.close()
    val header = lines.head.split(",").map(_.trim)
    val rows = lines.tail.map(_.split(",").map(_.trim))
    val predictors = header.indices.filter(_ != targetCol).toArray
    val x = MatrixD(rows.map(r => Array(1.0) ++ predictors.map(i => r(i).toDouble)))
    val y = VectorD(rows.map(r => r(targetCol).toDouble))
    val names = Array("intercept") ++ predictors.map(header(_))
    (x, y, names)

  def select(name: String, path: String, targetCol: Int): Unit =
    val (x, y, names) = loadCSV(path, targetCol)
    val model = new Regression(x, y, names)
    model.train(x, y)
    val (_, fullQof) = model.test(x, y)
    println("=" * 80)
    println(s"FORWARD FEATURE SELECTION: $name")
    println("Full model:")
    println(model.report(fullQof))

    // qk = 1 selects using adjusted R^2 in the ScalaTion Fit QoF vector.
    val (cols, qofSteps) = model.forwardSelAll(false)(using 1)
    println(s"Selected column indexes: $cols")
    println(s"Selected feature names: ${cols.toSeq.sorted.map(names(_)).mkString(", ")}")
    println("QoF by forward-selection step:")
    println(qofSteps)
    println(s"Best step: ${model.getBest}")

  @main def runForwardSelection(): Unit =
    // Two of the three datasets, as required by the rubric.
    select("AutoMPG", "data/auto_mpg_numeric.csv", 0)
    select("Concrete", "data/concrete.csv", 8)
