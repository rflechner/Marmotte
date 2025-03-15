namespace Marmotte

open System
open System.Net.Http
open System.Threading.Tasks
open Microsoft.AspNetCore.Builder
open Microsoft.AspNetCore.Http
open Microsoft.AspNetCore.Routing
open Microsoft.AspNetCore.Routing.Template
open Microsoft.FSharp.Core

[<RequireQualifiedAccess>]
module Http =

    type RequestHandler = HttpContext -> unit Task
        
    let map (verb: string) (routeTemplate: string) (handler: RequestHandler) (app: WebApplication) =
        let template = TemplateParser.Parse(routeTemplate)
        let func =
            Func<HttpContext, Task>(
                fun ctx -> ctx |> handler :> Task
            )
        let routeBuilder = app.MapMethods(routeTemplate, [verb], func)
        routeBuilder
    
    let get = map HttpMethods.Get

    let post = map HttpMethods.Post

    let delete = map HttpMethods.Delete

    let patch = map HttpMethods.Patch

    let put = map HttpMethods.Put

    let trace = map HttpMethods.Trace

    let options = map HttpMethods.Options
