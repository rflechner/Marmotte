namespace Marmotte

open Microsoft.AspNetCore.Builder

module RestfulDsl =

    type RestConfig = { OperationName: string option }
    
    type RestOperation =
        | Get of route: string * handler: obj * configure: (RestConfig -> RestConfig) option
        | Post of route: string * handler: obj * configure: (RestConfig -> RestConfig) option
        
    type RestResource =
        { Name: string; Operations: RestOperation list }
    
    type RestApiDefinition = RestResource list
        
    type RestfulBuilder(app: WebApplication) =
        
        member val ResourceName = "" with get, set
        
        member _.Yield(()) = app

        member _.Run _ = app
        
        [<CustomOperation("resource")>]
        member x.Resource (_:WebApplication, resourceName: string) =
            x.ResourceName <- resourceName
            app
        
        [<CustomOperation("get")>]
        member x.Map<'tin, 'tout, 'terr> (_:WebApplication, verb, routeTemplate: string, handler: Restful.ApiRequestHandler<'tin, 'tout, 'terr>) =
            Restful.map verb routeTemplate handler id { App=app; Name=x.ResourceName; Tags=x.ResourceName } |> ignore
            app
        
        [<CustomOperation("get")>]
        member x.Get<'tin, 'tout, 'terr> (_:WebApplication, routeTemplate: string, handler: Restful.ApiRequestHandler<'tin, 'tout, 'terr>) =
            Restful.get routeTemplate handler id { App=app; Name=x.ResourceName; Tags=x.ResourceName } |> ignore
            app
        
        [<CustomOperation("post")>]
        member x.Post<'tin, 'tout, 'terr> (_:WebApplication, routeTemplate: string, handler: Restful.ApiRequestHandler<'tin, 'tout, 'terr>) =
            Restful.post routeTemplate handler id { App=app; Name=x.ResourceName; Tags=x.ResourceName } |> ignore
            app
        
        [<CustomOperation("delete")>]
        member x.Delete<'tin, 'tout, 'terr> (_:WebApplication, routeTemplate: string, handler: Restful.ApiRequestHandler<'tin, 'tout, 'terr>) =
            Restful.delete routeTemplate handler id { App=app; Name=x.ResourceName; Tags=x.ResourceName } |> ignore
            app
        
        [<CustomOperation("patch")>]
        member x.Patch<'tin, 'tout, 'terr> (_:WebApplication, routeTemplate: string, handler: Restful.ApiRequestHandler<'tin, 'tout, 'terr>) =
            Restful.patch routeTemplate handler id { App=app; Name=x.ResourceName; Tags=x.ResourceName } |> ignore
            app
        
        [<CustomOperation("options")>]
        member x.Options<'tin, 'tout, 'terr> (_:WebApplication, routeTemplate: string, handler: Restful.ApiRequestHandler<'tin, 'tout, 'terr>) =
            Restful.options routeTemplate handler id { App=app; Name=x.ResourceName; Tags=x.ResourceName } |> ignore
            app
        
        [<CustomOperation("put")>]
        member x.Put<'tin, 'tout, 'terr> (_:WebApplication, routeTemplate: string, handler: Restful.ApiRequestHandler<'tin, 'tout, 'terr>) =
            Restful.put routeTemplate handler id { App=app; Name=x.ResourceName; Tags=x.ResourceName } |> ignore
            app
        
        [<CustomOperation("trace")>]
        member x.Trace<'tin, 'tout, 'terr> (_:WebApplication, routeTemplate: string, handler: Restful.ApiRequestHandler<'tin, 'tout, 'terr>) =
            Restful.trace routeTemplate handler id { App=app; Name=x.ResourceName; Tags=x.ResourceName } |> ignore
            app
    
    let restful app =
        let builder = RestfulBuilder(app)
        builder
    
    
